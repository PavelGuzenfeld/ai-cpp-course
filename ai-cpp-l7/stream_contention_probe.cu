// L7 exercise: what a neighbour's synchronisation choice costs you.
// Two "components" share one device. B is a latency-sensitive stage; A is a
// throughput stage. A's choice of sync call is the only thing that varies.
//
// Results and what they mean: README, "Your kernel is not the only thing on
// the device". Deliberately not in CMakeLists.txt -- the course image has no
// nvcc, so this builds against a host CUDA install.
//   nvcc -O2 -std=c++17 stream_contention_probe.cu -o probe
//
// A is sized to one block on purpose. Give A enough blocks to fill every SM
// and B queues on occupancy instead, which measures the scheduler rather than
// the sync policy and makes all three cases look identical.
#include <cstdio>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); std::exit(1);} } while(0)

__global__ void burn(float *d, int n, int iters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = d[i];
    for (int k = 0; k < iters; ++k) v = v * 1.000001f + 0.000001f;
    d[i] = v;
}

using Clock = std::chrono::steady_clock;

enum class Mode { BAlone, StreamScoped, DeviceWide, DefaultStream };

// A's sync policy is the independent variable; everything else is held fixed.
static void run_a(Mode mode, std::atomic<bool> &stop, float *buf, int n)
{
    cudaStream_t s = nullptr;
    if (mode != Mode::DefaultStream) CK(cudaStreamCreate(&s));
    while (!stop.load(std::memory_order_relaxed)) {
        burn<<<(n + 255) / 256, 256, 0, s>>>(buf, n, 400000);
        if (mode == Mode::DeviceWide) cudaDeviceSynchronize();
        else                          cudaStreamSynchronize(s);
    }
    if (s) cudaStreamDestroy(s);
}

struct Stats { double p50, p99, max; };

static Stats percentiles(std::vector<double> &v)
{
    std::sort(v.begin(), v.end());
    auto at = [&](double q) { return v[std::min(v.size() - 1, (size_t)(q * v.size()))]; };
    return {at(0.50), at(0.99), v.back()};
}

static Stats measure_b(Mode mode, int iters)
{
    int const nB = 1 << 12, nA = 256;   // A: 1 block, long-running -- leaves SMs free
    float *bufB = nullptr, *bufA = nullptr;
    CK(cudaMalloc(&bufB, nB * sizeof(float)));
    CK(cudaMalloc(&bufA, nA * sizeof(float)));
    cudaStream_t sB;
    CK(cudaStreamCreate(&sB));

    std::atomic<bool> stop{false};
    std::thread a;
    if (mode != Mode::BAlone) {
        a = std::thread(run_a, mode, std::ref(stop), bufA, nA);
        std::this_thread::sleep_for(std::chrono::milliseconds(200)); // let A saturate
    }

    for (int w = 0; w < 50; ++w) {
        burn<<<(nB + 255) / 256, 256, 0, sB>>>(bufB, nB, 50);
        cudaStreamSynchronize(sB);
    }

    std::vector<double> lat;
    lat.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = Clock::now();
        burn<<<(nB + 255) / 256, 256, 0, sB>>>(bufB, nB, 50);
        cudaStreamSynchronize(sB);
        lat.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }

    stop.store(true, std::memory_order_relaxed);
    if (a.joinable()) a.join();
    cudaStreamDestroy(sB);
    cudaFree(bufB);
    cudaFree(bufA);
    return percentiles(lat);
}

int main()
{
    cudaDeviceProp p{};
    CK(cudaGetDeviceProperties(&p, 0));
    std::printf("device: %s  SMs=%d  concurrentKernels=%d\n\n",
                p.name, p.multiProcessorCount, p.concurrentKernels);

    struct { Mode m; char const *label; } cases[] = {
        {Mode::BAlone,       "B alone (no neighbour)"},
        {Mode::StreamScoped, "A in own stream, cudaStreamSynchronize"},
        {Mode::DeviceWide,   "A in own stream, cudaDeviceSynchronize"},
        {Mode::DefaultStream,"A in the legacy default stream"},
    };

    std::printf("%-42s %9s %9s %9s\n", "case", "p50 (ms)", "p99 (ms)", "max (ms)");
    double base_p99 = 0.0;
    for (auto &c : cases) {
        Stats s = measure_b(c.m, 2000);
        if (c.m == Mode::BAlone) base_p99 = s.p99;
        std::printf("%-42s %9.4f %9.4f %9.4f", c.label, s.p50, s.p99, s.max);
        if (c.m != Mode::BAlone) std::printf("   p99 x%.2f vs alone", s.p99 / base_p99);
        std::printf("\n");
    }
    return 0;
}
