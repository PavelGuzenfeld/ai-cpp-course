#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "spsc_ring.hpp"

namespace nb = nanobind;

namespace
{
    constexpr std::size_t kRingCapacity = 256;

    template <typename Ring>
    std::vector<int> produce_and_consume(int n_items)
    {
        Ring ring;
        std::vector<int> received;
        received.reserve(static_cast<std::size_t>(n_items));

        std::jthread producer([&]
                               {
            for (int i = 0; i < n_items; ++i) {
                while (!ring.push(i)) {
                    std::this_thread::yield();
                }
            } });

        std::jthread consumer([&]
                               {
            int count = 0;
            while (count < n_items) {
                if (auto value = ring.try_pop()) {
                    received.push_back(*value);
                    ++count;
                } else {
                    std::this_thread::yield();
                }
            } });

        return received;
    }
} // namespace

// Runs a real producer/consumer pair over the correct SpscRing and returns
// what the consumer received. Releases the GIL: these are genuine OS
// threads and neither calls back into Python.
std::vector<int> run_correct_ring(int n_items)
{
    nb::gil_scoped_release release;
    return produce_and_consume<SpscRing<int, kRingCapacity>>(n_items);
}

// Same workload over the publish-before-write ring. Left to the caller to
// compare against `list(range(n_items))`.
std::vector<int> run_broken_ring(int n_items)
{
    nb::gil_scoped_release release;
    return produce_and_consume<BrokenSpscRing<int, kRingCapacity>>(n_items);
}

// Busy-waits for `ms` milliseconds, optionally releasing the GIL first.
// With the GIL held, no other Python bytecode can run for the duration of
// the call; with it released, a background Python thread can make real
// progress while this C++ code sleeps.
void busy_wait_ms(int ms, bool release_gil)
{
    if (release_gil)
    {
        nb::gil_scoped_release release;
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }
    else
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }
}

NB_MODULE(concurrency_native, m)
{
    m.def("run_correct_ring", &run_correct_ring, nb::arg("n_items"));
    m.def("run_broken_ring", &run_broken_ring, nb::arg("n_items"));
    m.def("busy_wait_ms", &busy_wait_ms, nb::arg("ms"), nb::arg("release_gil"));
}
