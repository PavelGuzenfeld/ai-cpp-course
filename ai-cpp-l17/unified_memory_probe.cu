// L17 Exercise 3 falsifier: "unified memory always removes the copy, so it
// will be faster than an explicit copy path." Orin NX / JP6, shared LPDDR.
// Verdict in verdict_unified_memory.md. Deliberately not wired into
// CMakeLists.txt: the course image has no nvcc, so this builds on the host.
//   /usr/local/cuda/bin/nvcc -O2 -std=c++17 unified_memory_probe.cu -o probe
#include <cstdio>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return 1;} } while(0)

__global__ void scale(float *d, size_t n, float k)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = d[i] * k + 1.0f;
}

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// One iteration = CPU fills the buffer, GPU scales it, CPU reads it back.
// That producer/consumer round trip is the thing the claim is about.
static double run_managed(size_t n, int iters)
{
    float *p = nullptr;
    if (cudaMallocManaged(&p, n * sizeof(float)) != cudaSuccess) return -1;
    int const threads = 256, blocks = (int)((n + threads - 1) / threads);
    for (int w = 0; w < 3; ++w) { scale<<<blocks, threads>>>(p, n, 1.01f); cudaDeviceSynchronize(); }
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) {
        for (size_t j = 0; j < n; j += 1024) p[j] = (float)i;
        scale<<<blocks, threads>>>(p, n, 1.01f);
        cudaDeviceSynchronize();
        volatile float sink = p[0]; (void)sink;
    }
    double t = ms_since(t0) / iters;
    cudaFree(p);
    return t;
}

static double run_explicit(size_t n, int iters)
{
    float *h = nullptr, *d = nullptr;
    if (cudaHostAlloc(&h, n * sizeof(float), cudaHostAllocDefault) != cudaSuccess) return -1;
    if (cudaMalloc(&d, n * sizeof(float)) != cudaSuccess) { cudaFreeHost(h); return -1; }
    int const threads = 256, blocks = (int)((n + threads - 1) / threads);
    for (int w = 0; w < 3; ++w) { scale<<<blocks, threads>>>(d, n, 1.01f); cudaDeviceSynchronize(); }
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) {
        for (size_t j = 0; j < n; j += 1024) h[j] = (float)i;
        cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice);
        scale<<<blocks, threads>>>(d, n, 1.01f);
        cudaMemcpy(h, d, n * sizeof(float), cudaMemcpyDeviceToHost);
        volatile float sink = h[0]; (void)sink;
    }
    double t = ms_since(t0) / iters;
    cudaFree(d); cudaFreeHost(h);
    return t;
}

int main()
{
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s  integrated=%d  canMapHostMemory=%d  managedMemory=%d\n",
                prop.name, prop.integrated, prop.canMapHostMemory, prop.managedMemory);
    std::printf("%12s %14s %14s %10s  %s\n", "elems", "managed(ms)", "explicit(ms)", "ratio", "verdict");
    for (size_t n : {1u<<12, 1u<<16, 1u<<20, 1u<<22, 1u<<24}) {
        int iters = n >= (1u<<22) ? 50 : 200;
        double m = run_managed(n, iters), e = run_explicit(n, iters);
        if (m < 0 || e < 0) { std::printf("%12zu  alloc failed\n", n); continue; }
        std::printf("%12zu %14.4f %14.4f %10.2f  %s\n", n, m, e, m / e,
                    m <= e ? "managed faster-or-equal" : "CLAIM FALSIFIED: explicit faster");
    }
    return 0;
}
