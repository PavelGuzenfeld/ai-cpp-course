/**
 * CUDA IPC Consumer — opens IPC handles from the producer and reads GPU memory
 * directly, without any CPU↔GPU copy.
 *
 * Build:
 *   nvcc -o cuda_ipc_consumer cuda_ipc_consumer.cu -O2
 *
 * Run: (after cuda_ipc_producer is running)
 *   ./cuda_ipc_consumer
 */

#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <chrono>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

struct IpcHandles {
    cudaIpcMemHandle_t mem_handle;
    cudaIpcEventHandle_t event_handle;
    int num_elements;
    int ready;
};

// Kernel: sum reduction (for verification)
__global__ void partial_sum(const float* data, double* result, int n)
{
    __shared__ double sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? static_cast<double>(data[idx]) : 0.0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(result, sdata[0]);
}

// Kernel: verify pattern (returns count of mismatches)
__global__ void verify_pattern(const float* data, int n, float base, int* mismatches)
{
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float expected = base + static_cast<float>(i);
        if (fabsf(data[i] - expected) > 0.001f)
            atomicAdd(mismatches, 1);
    }
}

int main()
{
    constexpr const char* SHM_NAME = "/cuda_ipc_demo";

    // 1. Open shared memory and wait for producer
    int fd = shm_open(SHM_NAME, O_RDONLY, 0666);
    if (fd < 0) {
        fprintf(stderr, "Producer not running. Start cuda_ipc_producer first.\n");
        return 1;
    }

    auto* handles = static_cast<const IpcHandles*>(
        mmap(nullptr, sizeof(IpcHandles), PROT_READ, MAP_SHARED, fd, 0));

    while (!handles->ready) { usleep(1000); }
    __sync_synchronize();

    int N = handles->num_elements;
    printf("=== CUDA IPC Consumer ===\n");
    printf("Received handle for %d elements (%.1f MB)\n", N, N * sizeof(float) / 1e6);

    // 2. Open IPC memory handle — maps producer's GPU memory into this process
    float* d_data = nullptr;
    auto t0 = std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&d_data),
                                     handles->mem_handle,
                                     cudaIpcMemLazyEnablePeerAccess));
    auto t1 = std::chrono::high_resolution_clock::now();
    float open_us = std::chrono::duration<float, std::micro>(t1 - t0).count();

    // 3. Open IPC event and synchronize
    cudaEvent_t event;
    cudaIpcEventHandle_t evt_handle;
    memcpy(&evt_handle, &handles->event_handle, sizeof(evt_handle));
    CHECK_CUDA(cudaIpcOpenEventHandle(&event, evt_handle));
    CHECK_CUDA(cudaStreamWaitEvent(nullptr, event));

    printf("IPC memory opened in %.1f us (zero-copy, no CPU transfer)\n", open_us);

    // 4. Verify data directly on GPU — no D2H copy needed
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    int* d_mismatches;
    CHECK_CUDA(cudaMalloc(&d_mismatches, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_mismatches, 0, sizeof(int)));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    verify_pattern<<<blocks, threads>>>(d_data, N, 1000.0f, d_mismatches);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float verify_ms;
    CHECK_CUDA(cudaEventElapsedTime(&verify_ms, start, stop));

    int mismatches;
    CHECK_CUDA(cudaMemcpy(&mismatches, d_mismatches, sizeof(int), cudaMemcpyDeviceToHost));

    printf("GPU verification: %s (%d mismatches, %.3f ms)\n",
           mismatches == 0 ? "PASS" : "FAIL", mismatches, verify_ms);

    // 5. Benchmark: compare IPC access vs copy-through-CPU
    // IPC path: just read GPU memory (already mapped)
    double* d_sum;
    CHECK_CUDA(cudaMalloc(&d_sum, sizeof(double)));
    CHECK_CUDA(cudaMemset(d_sum, 0, sizeof(double)));

    CHECK_CUDA(cudaEventRecord(start));
    partial_sum<<<blocks, threads>>>(d_data, d_sum, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ipc_ms;
    CHECK_CUDA(cudaEventElapsedTime(&ipc_ms, start, stop));

    // Copy-through-CPU path: D2H + H2D + compute
    float* h_buf;
    float* d_copy;
    CHECK_CUDA(cudaMallocHost(&h_buf, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_copy, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_sum, 0, sizeof(double)));

    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpy(h_buf, d_data, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(d_copy, h_buf, N * sizeof(float), cudaMemcpyHostToDevice));
    partial_sum<<<blocks, threads>>>(d_copy, d_sum, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float copy_ms;
    CHECK_CUDA(cudaEventElapsedTime(&copy_ms, start, stop));

    printf("\n=== IPC vs Copy-Through-CPU Benchmark ===\n");
    printf("IPC (direct GPU access):     %.3f ms\n", ipc_ms);
    printf("Copy-through-CPU (D2H+H2D):  %.3f ms\n", copy_ms);
    printf("Speedup:                     %.1fx\n", copy_ms / ipc_ms);

    // Cleanup
    CHECK_CUDA(cudaIpcCloseMemHandle(d_data));
    CHECK_CUDA(cudaFree(d_mismatches));
    CHECK_CUDA(cudaFree(d_sum));
    CHECK_CUDA(cudaFree(d_copy));
    CHECK_CUDA(cudaFreeHost(h_buf));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(event));
    munmap(const_cast<IpcHandles*>(handles), sizeof(IpcHandles));
    close(fd);

    return 0;
}
