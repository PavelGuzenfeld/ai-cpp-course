/**
 * unified-memory-demo.cu
 *
 * Demonstrates that unified memory (cudaMallocManaged) performs differently
 * on Jetson vs desktop GPUs.
 *
 * On desktop (discrete GPU):
 *   - Unified memory triggers page faults when the GPU first accesses
 *     CPU-allocated pages. The driver migrates pages across PCIe on demand.
 *   - Explicit allocation (cudaMalloc + cudaMemcpy) avoids page faults by
 *     transferring all data upfront via DMA.
 *   - Result: explicit is typically 2-10x faster for first-touch workloads.
 *
 * On Jetson (integrated GPU, shared LPDDR):
 *   - Unified memory maps to the same physical DRAM. No page migration
 *     is needed because CPU and GPU share the memory bus.
 *   - cudaMallocManaged pointers are zero-copy: both CPU and GPU access
 *     the same physical pages directly.
 *   - Result: unified memory matches or slightly beats explicit allocation
 *     (which still pays cudaMemcpy dispatch overhead for no benefit).
 *
 * Build:
 *   nvcc -O2 -o unified-memory-demo unified-memory-demo.cu
 *
 * Run:
 *   ./unified-memory-demo
 *   ./unified-memory-demo 67108864    # custom element count (64M floats = 256 MB)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Simple vector-add kernel (memory-bandwidth bound)
// ---------------------------------------------------------------------------

__global__ void vector_add(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c,
                           int n)
{
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// ---------------------------------------------------------------------------
// Initialization kernel (fill arrays on GPU to avoid first-touch on CPU)
// ---------------------------------------------------------------------------

__global__ void fill_array(float* arr, float value, int n)
{
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        arr[i] = value + static_cast<float>(i % 1000) * 0.001f;
    }
}

// ---------------------------------------------------------------------------
// Detect Jetson
// ---------------------------------------------------------------------------

static bool is_jetson()
{
    FILE* f = fopen("/proc/device-tree/model", "r");
    if (!f) return false;

    char buf[256] = {};
    size_t nread = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);

    if (nread == 0) return false;

    // Convert to lowercase for matching
    for (size_t i = 0; i < nread; ++i) {
        if (buf[i] >= 'A' && buf[i] <= 'Z') buf[i] += 32;
    }

    return strstr(buf, "jetson") != nullptr || strstr(buf, "nvidia") != nullptr;
}

// ---------------------------------------------------------------------------
// Benchmark: Explicit allocation (cudaMalloc + cudaMemcpy)
// ---------------------------------------------------------------------------

static float bench_explicit(int n, int n_iters)
{
    size_t bytes = static_cast<size_t>(n) * sizeof(float);

    // Host arrays (pinned for optimal transfer)
    float* h_a;
    float* h_b;
    CUDA_CHECK(cudaMallocHost(&h_a, bytes));
    CUDA_CHECK(cudaMallocHost(&h_b, bytes));

    // Initialize on host
    for (int i = 0; i < n; ++i) {
        h_a[i] = 1.0f + static_cast<float>(i % 1000) * 0.001f;
        h_b[i] = 2.0f + static_cast<float>(i % 1000) * 0.001f;
    }

    // Device arrays
    float* d_a;
    float* d_b;
    float* d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    // Warmup: transfer + compute
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed iterations (transfer + compute each time)
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < n_iters; ++iter) {
        CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
        vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float total_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_b));

    return total_ms / static_cast<float>(n_iters);
}

// ---------------------------------------------------------------------------
// Benchmark: Explicit, compute-only (no transfer -- pre-resident on GPU)
// ---------------------------------------------------------------------------

static float bench_explicit_compute_only(int n, int n_iters)
{
    size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float* d_a;
    float* d_b;
    float* d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    // Initialize on GPU (no host transfer)
    fill_array<<<blocks, threads>>>(d_a, 1.0f, n);
    fill_array<<<blocks, threads>>>(d_b, 2.0f, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Warmup
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < n_iters; ++iter) {
        vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float total_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return total_ms / static_cast<float>(n_iters);
}

// ---------------------------------------------------------------------------
// Benchmark: Unified memory (cudaMallocManaged)
// ---------------------------------------------------------------------------

static float bench_unified(int n, int n_iters)
{
    size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float* a;
    float* b;
    float* c;
    CUDA_CHECK(cudaMallocManaged(&a, bytes));
    CUDA_CHECK(cudaMallocManaged(&b, bytes));
    CUDA_CHECK(cudaMallocManaged(&c, bytes));

    // Initialize on CPU (this is the typical unified memory pattern:
    // CPU writes, then GPU reads)
    for (int i = 0; i < n; ++i) {
        a[i] = 1.0f + static_cast<float>(i % 1000) * 0.001f;
        b[i] = 2.0f + static_cast<float>(i % 1000) * 0.001f;
    }

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    // Warmup: first GPU access triggers page migration on desktop
    vector_add<<<blocks, threads>>>(a, b, c, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Re-initialize on CPU to force migration again on desktop.
    // On Jetson this is a no-op (same physical pages).
    for (int i = 0; i < n; ++i) {
        a[i] = 1.0f + static_cast<float>(i % 1000) * 0.001f;
        b[i] = 2.0f + static_cast<float>(i % 1000) * 0.001f;
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < n_iters; ++iter) {
        // On desktop: GPU access triggers page faults + PCIe migration
        // On Jetson: GPU accesses the same LPDDR pages directly
        vector_add<<<blocks, threads>>>(a, b, c, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Touch on CPU to invalidate GPU caches / force re-migration on desktop
        // On Jetson: this is just a regular memory write, no migration
        c[0] = 0.0f;
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float total_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));

    return total_ms / static_cast<float>(n_iters);
}

// ---------------------------------------------------------------------------
// Benchmark: Unified memory, GPU-initialized (no CPU first-touch)
// ---------------------------------------------------------------------------

static float bench_unified_gpu_init(int n, int n_iters)
{
    size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float* a;
    float* b;
    float* c;
    CUDA_CHECK(cudaMallocManaged(&a, bytes));
    CUDA_CHECK(cudaMallocManaged(&b, bytes));
    CUDA_CHECK(cudaMallocManaged(&c, bytes));

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    // Initialize on GPU -- pages are first-touched by GPU
    fill_array<<<blocks, threads>>>(a, 1.0f, n);
    fill_array<<<blocks, threads>>>(b, 2.0f, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Warmup
    vector_add<<<blocks, threads>>>(a, b, c, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < n_iters; ++iter) {
        vector_add<<<blocks, threads>>>(a, b, c, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float total_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));

    return total_ms / static_cast<float>(n_iters);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    int n = 4 * 1024 * 1024;  // 4M floats = 16 MB per array
    if (argc > 1) {
        n = atoi(argv[1]);
        if (n <= 0) {
            fprintf(stderr, "Usage: %s [num_elements]\n", argv[0]);
            return 1;
        }
    }

    int n_iters = 20;
    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float size_mb = static_cast<float>(bytes) / (1024.0f * 1024.0f);
    bool on_jetson = is_jetson();

    // Print device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("============================================================\n");
    printf("UNIFIED MEMORY vs EXPLICIT ALLOCATION\n");
    printf("============================================================\n");
    printf("  Device:     %s\n", prop.name);
    printf("  Platform:   %s\n", on_jetson ? "Jetson (integrated GPU, shared memory)"
                                            : "Desktop (discrete GPU, separate memory)");
    printf("  SM count:   %d\n", prop.multiProcessorCount);
    printf("  Array size: %d elements (%.1f MB per array, %.1f MB total)\n",
           n, size_mb, size_mb * 3);
    printf("  Iterations: %d\n", n_iters);
    printf("\n");

    // Run benchmarks
    float t_explicit_xfer  = bench_explicit(n, n_iters);
    float t_explicit_only  = bench_explicit_compute_only(n, n_iters);
    float t_unified_cpu    = bench_unified(n, n_iters);
    float t_unified_gpu    = bench_unified_gpu_init(n, n_iters);

    // Print results
    printf("%-45s %8.3f ms\n", "Explicit (cudaMalloc + cudaMemcpy + compute)", t_explicit_xfer);
    printf("%-45s %8.3f ms\n", "Explicit (compute only, data pre-resident)", t_explicit_only);
    printf("%-45s %8.3f ms\n", "Unified (CPU-initialized, GPU compute)", t_unified_cpu);
    printf("%-45s %8.3f ms\n", "Unified (GPU-initialized, GPU compute)", t_unified_gpu);

    printf("\n");
    printf("Ratios (lower = faster):\n");
    printf("  Unified (CPU-init) / Explicit (compute-only): %.2fx\n",
           t_unified_cpu / t_explicit_only);
    printf("  Unified (GPU-init) / Explicit (compute-only): %.2fx\n",
           t_unified_gpu / t_explicit_only);
    printf("  Explicit (with transfer) / Explicit (compute-only): %.2fx\n",
           t_explicit_xfer / t_explicit_only);

    printf("\n");
    if (on_jetson) {
        printf("JETSON ANALYSIS:\n");
        printf("  On Jetson, unified memory (GPU-initialized) should be very close\n");
        printf("  to explicit compute-only, because both access the same physical\n");
        printf("  LPDDR. The CPU-initialized case may be slightly slower due to\n");
        printf("  cache coherency traffic, but there is no page fault overhead.\n");
        printf("\n");
        printf("  If unified (CPU-init) is more than 1.5x slower than explicit,\n");
        printf("  check that the GPU is not thermally throttled (run tegrastats).\n");
    } else {
        printf("DESKTOP ANALYSIS:\n");
        printf("  On desktop, unified memory with CPU initialization is expected\n");
        printf("  to be significantly slower than explicit allocation because each\n");
        printf("  GPU access triggers page faults and PCIe page migration.\n");
        printf("\n");
        printf("  Unified with GPU initialization should be close to explicit\n");
        printf("  compute-only, because pages are already resident on the GPU.\n");
        printf("\n");
        printf("  The explicit (with transfer) cost shows the cudaMemcpy overhead\n");
        printf("  that CUDA IPC eliminates in multi-process pipelines.\n");
        printf("\n");
        printf("  On Jetson, you would see unified (CPU-init) close to 1.0x because\n");
        printf("  there is no PCIe bus -- CPU and GPU share the same physical memory.\n");
    }

    return 0;
}
