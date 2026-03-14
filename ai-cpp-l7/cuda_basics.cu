/**
 * CUDA Basics — grid-stride loops, unified memory, and kernel fundamentals.
 *
 * Build:
 *   nvcc -o cuda_basics cuda_basics.cu -O2
 * Run:
 *   ./cuda_basics
 */

#include <cstdio>
#include <cstdlib>
#include <chrono>

// ---------------------------------------------------------------------------
// Grid-stride loop: each thread processes multiple elements, works for any N
// ---------------------------------------------------------------------------
__global__ void vector_add(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c,
                           int n)
{
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride)
        c[i] = a[i] + b[i];
}

// ---------------------------------------------------------------------------
// Fused multiply-add with scaling — shows combining operations in one kernel
// ---------------------------------------------------------------------------
__global__ void fused_scale_add(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                float scale_a, float scale_b,
                                int n)
{
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride)
        c[i] = scale_a * a[i] + scale_b * b[i];
}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------
void vector_add_cpu(const float* a, const float* b, float* c, int n)
{
    for (int i = 0; i < n; ++i)
        c[i] = a[i] + b[i];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
void fill_random(float* p, int n)
{
    for (int i = 0; i < n; ++i)
        p[i] = static_cast<float>(rand()) / RAND_MAX;
}

bool verify(const float* ref, const float* test, int n, float tol = 1e-5f)
{
    for (int i = 0; i < n; ++i)
        if (fabsf(ref[i] - test[i]) > tol) return false;
    return true;
}

int main()
{
    constexpr int N = 1 << 22;  // ~4M elements
    constexpr size_t bytes = N * sizeof(float);

    // --- Unified Memory path ---
    float *ua, *ub, *uc;
    cudaMallocManaged(&ua, bytes);
    cudaMallocManaged(&ub, bytes);
    cudaMallocManaged(&uc, bytes);

    fill_random(ua, N);
    fill_random(ub, N);

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    // Warm up
    vector_add<<<blocks, threads>>>(ua, ub, uc, N);
    cudaDeviceSynchronize();

    // Timed GPU (unified memory)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vector_add<<<blocks, threads>>>(ua, ub, uc, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_unified_ms = 0;
    cudaEventElapsedTime(&gpu_unified_ms, start, stop);

    // --- Explicit memory path (pinned host + device) ---
    float *ha, *hb, *hc_gpu;
    cudaMallocHost(&ha, bytes);
    cudaMallocHost(&hb, bytes);
    cudaMallocHost(&hc_gpu, bytes);

    float *da, *db, *dc;
    cudaMalloc(&da, bytes);
    cudaMalloc(&db, bytes);
    cudaMalloc(&dc, bytes);

    memcpy(ha, ua, bytes);
    memcpy(hb, ub, bytes);

    // Warm up
    cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice);
    vector_add<<<blocks, threads>>>(da, db, dc, N);
    cudaMemcpy(hc_gpu, dc, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice);
    vector_add<<<blocks, threads>>>(da, db, dc, N);
    cudaMemcpy(hc_gpu, dc, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_explicit_ms = 0;
    cudaEventElapsedTime(&gpu_explicit_ms, start, stop);

    // --- CPU reference ---
    float* hc_cpu = (float*)malloc(bytes);

    auto t0 = std::chrono::high_resolution_clock::now();
    vector_add_cpu(ha, hb, hc_cpu, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    float cpu_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // --- Verify ---
    bool ok = verify(hc_cpu, hc_gpu, N);

    printf("=== Vector Add (%d elements, %.1f MB) ===\n", N, bytes / 1e6);
    printf("CPU:                 %.3f ms\n", cpu_ms);
    printf("GPU (unified mem):   %.3f ms\n", gpu_unified_ms);
    printf("GPU (pinned+copy):   %.3f ms  (includes H2D + D2H)\n", gpu_explicit_ms);
    printf("Speedup (vs CPU):    %.1fx (unified), %.1fx (explicit)\n",
           cpu_ms / gpu_unified_ms, cpu_ms / gpu_explicit_ms);
    printf("Verify: %s\n", ok ? "PASS" : "FAIL");

    // --- Fused kernel demo ---
    cudaEventRecord(start);
    fused_scale_add<<<blocks, threads>>>(da, db, dc, 0.5f, 2.0f, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float fused_ms = 0;
    cudaEventElapsedTime(&fused_ms, start, stop);
    printf("\nFused scale_add kernel: %.3f ms (kernel only)\n", fused_ms);

    // Cleanup
    cudaFree(da); cudaFree(db); cudaFree(dc);
    cudaFreeHost(ha); cudaFreeHost(hb); cudaFreeHost(hc_gpu);
    cudaFree(ua); cudaFree(ub); cudaFree(uc);
    free(hc_cpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ok ? 0 : 1;
}
