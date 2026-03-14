/**
 * CUDA IPC Producer — allocates GPU memory, fills it, and exports IPC handles
 * for a consumer process to read without any CPU round-trip.
 *
 * This demonstrates GPU-side inter-process communication: two processes share
 * device memory directly, avoiding the CPU↔GPU copy chain entirely.
 *
 * Build:
 *   nvcc -o cuda_ipc_producer cuda_ipc_producer.cu -O2
 *   nvcc -o cuda_ipc_consumer cuda_ipc_consumer.cu -O2
 *
 * Run (two terminals):
 *   Terminal 1: ./cuda_ipc_producer
 *   Terminal 2: ./cuda_ipc_consumer    (after producer prints "ready")
 */

#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Shared region layout (passed via POSIX shm between processes)
struct IpcHandles {
    cudaIpcMemHandle_t mem_handle;
    cudaIpcEventHandle_t event_handle;
    int num_elements;
    int ready;  // flag: 1 = handles are valid
};

// Simple kernel: fill buffer with a pattern
__global__ void fill_pattern(float* data, int n, float base)
{
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride)
        data[i] = base + static_cast<float>(i);
}

int main()
{
    constexpr int N = 1 << 20;  // 1M floats = 4MB
    constexpr size_t bytes = N * sizeof(float);
    constexpr const char* SHM_NAME = "/cuda_ipc_demo";

    // 1. Allocate device memory
    float* d_data = nullptr;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));

    // 2. Fill with a known pattern on GPU
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    fill_pattern<<<blocks, threads>>>(d_data, N, 1000.0f);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 3. Create a CUDA event to signal when data is ready
    cudaEvent_t event;
    CHECK_CUDA(cudaEventCreate(&event, cudaEventInterprocess | cudaEventDisableTimingPeer));
    CHECK_CUDA(cudaEventRecord(event));
    CHECK_CUDA(cudaEventSynchronize(event));

    // 4. Get IPC handles
    cudaIpcMemHandle_t mem_handle;
    cudaIpcEventHandle_t event_handle;
    CHECK_CUDA(cudaIpcGetMemHandle(&mem_handle, d_data));
    CHECK_CUDA(cudaIpcGetEventHandle(&event_handle, event));

    // 5. Share handles via POSIX shared memory
    int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (fd < 0) { perror("shm_open"); return 1; }
    ftruncate(fd, sizeof(IpcHandles));

    auto* handles = static_cast<IpcHandles*>(
        mmap(nullptr, sizeof(IpcHandles), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));

    memcpy(&handles->mem_handle, &mem_handle, sizeof(mem_handle));
    memcpy(&handles->event_handle, &event_handle, sizeof(event_handle));
    handles->num_elements = N;
    __sync_synchronize();
    handles->ready = 1;

    printf("=== CUDA IPC Producer ===\n");
    printf("Allocated %.1f MB on GPU\n", bytes / 1e6);
    printf("Filled with pattern: data[i] = 1000.0 + i\n");
    printf("IPC handles exported to %s\n", SHM_NAME);
    printf("Waiting for consumer... (press Enter to cleanup)\n");
    getchar();

    // Cleanup
    munmap(handles, sizeof(IpcHandles));
    close(fd);
    shm_unlink(SHM_NAME);
    CHECK_CUDA(cudaEventDestroy(event));
    CHECK_CUDA(cudaFree(d_data));

    printf("Cleaned up.\n");
    return 0;
}
