#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
#include <algorithm>

namespace nb = nanobind;

// Pointer-chasing node for random access benchmark
struct Node
{
    Node *next;
    char padding[56]; // pad to 64 bytes (one cache line)
};

static_assert(sizeof(Node) == 64, "Node must be exactly one cache line");

struct BenchmarkResult
{
    std::vector<int64_t> sizes_bytes;
    std::vector<double> sequential_ns_per_access;
    std::vector<double> random_ns_per_access;
};

struct StrideBenchmarkResult
{
    std::vector<int64_t> strides;
    std::vector<double> ns_per_access;
};

// Sequential access benchmark: iterate through array linearly
double benchmark_sequential(size_t array_size_bytes, int iterations)
{
    size_t count = array_size_bytes / sizeof(int32_t);
    if (count == 0)
        count = 1;

    std::vector<int32_t> data(count);
    std::iota(data.begin(), data.end(), 0);

    // Warm up cache
    volatile int32_t sink = 0;
    for (size_t i = 0; i < count; ++i)
    {
        sink += data[i];
    }

    auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
    {
        for (size_t i = 0; i < count; ++i)
        {
            sink += data[i];
        }
    }
    auto end = std::chrono::steady_clock::now();

    double total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double ns_per_access = total_ns / (static_cast<double>(count) * iterations);
    return ns_per_access;
}

// Random access benchmark using pointer chasing
// This defeats hardware prefetching and measures true memory latency
double benchmark_random(size_t array_size_bytes, int iterations)
{
    size_t node_count = array_size_bytes / sizeof(Node);
    if (node_count < 2)
        node_count = 2;

    std::vector<Node> nodes(node_count);

    // Create a random permutation for pointer chasing
    std::vector<size_t> indices(node_count);
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(42); // fixed seed for reproducibility
    std::shuffle(indices.begin() + 1, indices.end(), rng);

    // Link nodes in shuffled order
    for (size_t i = 0; i < node_count - 1; ++i)
    {
        nodes[indices[i]].next = &nodes[indices[i + 1]];
    }
    nodes[indices[node_count - 1]].next = &nodes[indices[0]]; // circular

    // Warm up
    Node *current = &nodes[indices[0]];
    for (size_t i = 0; i < node_count; ++i)
    {
        current = current->next;
    }

    size_t accesses_per_iter = node_count;
    auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
    {
        for (size_t i = 0; i < accesses_per_iter; ++i)
        {
            current = current->next;
        }
    }
    auto end = std::chrono::steady_clock::now();

    // Prevent optimization of the pointer chase
    volatile auto prevent_opt = current;
    (void)prevent_opt;

    double total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double ns_per_access = total_ns / (static_cast<double>(accesses_per_iter) * iterations);
    return ns_per_access;
}

// Run benchmarks across array sizes from 4KB to 256MB
BenchmarkResult run_cache_benchmark(int iterations = 10)
{
    BenchmarkResult result;

    // Sizes: 4KB, 8KB, 16KB, 32KB, 64KB, 128KB, 256KB, 512KB,
    //        1MB, 2MB, 4MB, 8MB, 16MB, 32MB, 64MB, 128MB, 256MB
    for (int64_t size = 4 * 1024; size <= 256 * 1024 * 1024LL; size *= 2)
    {
        result.sizes_bytes.push_back(size);

        // Adjust iterations for larger sizes to keep runtime reasonable
        int iters = iterations;
        if (size > 16 * 1024 * 1024)
            iters = std::max(1, iterations / 4);
        else if (size > 1024 * 1024)
            iters = std::max(1, iterations / 2);

        double seq_ns = benchmark_sequential(size, iters);
        double rand_ns = benchmark_random(size, iters);

        result.sequential_ns_per_access.push_back(seq_ns);
        result.random_ns_per_access.push_back(rand_ns);
    }

    return result;
}

// Stride benchmark: fixed array size, vary stride
StrideBenchmarkResult run_stride_benchmark(int64_t array_size_bytes = 64 * 1024 * 1024,
                                           int iterations = 5)
{
    StrideBenchmarkResult result;

    size_t count = array_size_bytes / sizeof(int32_t);
    std::vector<int32_t> data(count);
    std::iota(data.begin(), data.end(), 0);

    // Strides: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 bytes
    // Converted to int32_t stride: divide by sizeof(int32_t)
    for (int64_t stride_bytes = 1; stride_bytes <= 4096; stride_bytes *= 2)
    {
        size_t stride_elements = std::max<size_t>(1, stride_bytes / sizeof(int32_t));

        volatile int32_t sink = 0;

        // Warm up
        for (size_t i = 0; i < count; i += stride_elements)
        {
            sink += data[i];
        }

        auto start = std::chrono::steady_clock::now();
        for (int iter = 0; iter < iterations; ++iter)
        {
            for (size_t i = 0; i < count; i += stride_elements)
            {
                sink += data[i];
            }
        }
        auto end = std::chrono::steady_clock::now();

        size_t accesses = 0;
        for (size_t i = 0; i < count; i += stride_elements)
            accesses++;

        double total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        double ns_per_access = total_ns / (static_cast<double>(accesses) * iterations);

        result.strides.push_back(stride_bytes);
        result.ns_per_access.push_back(ns_per_access);
    }

    return result;
}

NB_MODULE(cache_benchmark, m)
{
    m.doc() = "Cache hierarchy benchmark — reveals L1/L2/L3/RAM boundaries";

    nb::class_<BenchmarkResult>(m, "BenchmarkResult")
        .def_ro("sizes_bytes", &BenchmarkResult::sizes_bytes)
        .def_ro("sequential_ns_per_access", &BenchmarkResult::sequential_ns_per_access)
        .def_ro("random_ns_per_access", &BenchmarkResult::random_ns_per_access);

    nb::class_<StrideBenchmarkResult>(m, "StrideBenchmarkResult")
        .def_ro("strides", &StrideBenchmarkResult::strides)
        .def_ro("ns_per_access", &StrideBenchmarkResult::ns_per_access);

    m.def("benchmark_sequential", &benchmark_sequential,
          nb::arg("array_size_bytes"), nb::arg("iterations") = 10,
          "Measure sequential access latency (ns/access) for a given array size");

    m.def("benchmark_random", &benchmark_random,
          nb::arg("array_size_bytes"), nb::arg("iterations") = 10,
          "Measure random access latency (ns/access) using pointer chasing");

    m.def("run_cache_benchmark", &run_cache_benchmark,
          nb::arg("iterations") = 10,
          "Run sequential and random access benchmarks across sizes 4KB-256MB");

    m.def("run_stride_benchmark", &run_stride_benchmark,
          nb::arg("array_size_bytes") = 64 * 1024 * 1024,
          nb::arg("iterations") = 5,
          "Run stride access benchmark with varying stride sizes");
}
