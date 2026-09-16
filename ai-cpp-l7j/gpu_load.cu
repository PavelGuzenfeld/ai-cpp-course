// A neighbour that keeps the GPU busy, for engine_contention_bench.sh.
// Not a benchmark -- it produces no numbers, it only occupies the device.
//   nvcc -O2 gpu_load.cu -o /tmp/gpu_load
#include <cuda_runtime.h>

__global__ void burn(float *d, int n, int iters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = d[i];
    for (int k = 0; k < iters; ++k) v = v * 1.000001f + 1e-6f;
    d[i] = v;
}

int main()
{
    int const n = 1 << 22;
    float *d = nullptr;
    if (cudaMalloc(&d, n * sizeof(float)) != cudaSuccess) return 1;
    for (;;)
    {
        burn<<<(n + 255) / 256, 256>>>(d, n, 2000);
        cudaDeviceSynchronize();
    }
}
