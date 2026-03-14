#include <span>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <chrono>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>

namespace nb = nanobind;

// ============================================================================
// std::span — Safe Views Without Copies
// ============================================================================

// Sum elements using std::span (safe, knows its own size)
static double span_sum(const std::vector<double>& data) {
    std::span<const double> view(data);
    double total = 0.0;
    for (double val : view) {
        total += val;
    }
    return total;
}

// Return a sub-span (zero-cost view) — we copy the slice out for Python
static std::vector<double> span_slice(const std::vector<double>& data,
                                       size_t offset, size_t count) {
    std::span<const double> view(data);
    if (offset + count > view.size()) {
        throw std::out_of_range("span_slice: offset + count exceeds span size");
    }
    auto sub = view.subspan(offset, count);
    return std::vector<double>(sub.begin(), sub.end());
}

// Bounds-checked element access (raises on out-of-bounds)
static double safe_at(const std::vector<double>& data, size_t index) {
    std::span<const double> view(data);
    if (index >= view.size()) {
        throw std::out_of_range("safe_at: index " + std::to_string(index) +
                                " out of range for span of size " +
                                std::to_string(view.size()));
    }
    return view[index];
}

// Raw pointer version for comparison — same logic, no safety
static double raw_pointer_sum(const std::vector<double>& data) {
    const double* ptr = data.data();
    size_t n = data.size();
    double total = 0.0;
    for (size_t i = 0; i < n; ++i) {
        total += ptr[i];
    }
    return total;
}

// Benchmark: measure time for N iterations of span_sum vs raw_pointer_sum
// Returns (span_time_us, raw_time_us) in microseconds
static std::pair<double, double> benchmark_span_vs_raw(
        const std::vector<double>& data, int iterations) {
    using clock = std::chrono::high_resolution_clock;

    // Warm up
    volatile double sink = 0.0;
    sink = span_sum(data);
    sink = raw_pointer_sum(data);
    (void)sink;

    // Benchmark span
    auto t0 = clock::now();
    for (int i = 0; i < iterations; ++i) {
        sink = span_sum(data);
    }
    auto t1 = clock::now();
    double span_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    // Benchmark raw pointer
    auto t2 = clock::now();
    for (int i = 0; i < iterations; ++i) {
        sink = raw_pointer_sum(data);
    }
    auto t3 = clock::now();
    double raw_us = std::chrono::duration<double, std::micro>(t3 - t2).count();

    return {span_us, raw_us};
}

// ============================================================================
// nanobind bindings
// ============================================================================

NB_MODULE(safe_views, m) {
    m.doc() = "Lesson 11: std::span — safe views without copies";

    m.def("span_sum", &span_sum,
          nb::arg("data"),
          "Sum elements using std::span (safe, bounds-aware).");

    m.def("span_slice", &span_slice,
          nb::arg("data"), nb::arg("offset"), nb::arg("count"),
          "Return a slice of the data (zero-cost span internally, copied to "
          "Python list). Raises IndexError on out-of-bounds.");

    m.def("safe_at", &safe_at,
          nb::arg("data"), nb::arg("index"),
          "Bounds-checked element access. Raises IndexError on out-of-bounds.");

    m.def("raw_pointer_sum", &raw_pointer_sum,
          nb::arg("data"),
          "Sum elements using raw pointer (unsafe, for comparison).");

    m.def("benchmark_span_vs_raw", &benchmark_span_vs_raw,
          nb::arg("data"), nb::arg("iterations"),
          "Benchmark span_sum vs raw_pointer_sum. "
          "Returns (span_time_us, raw_time_us).");
}
