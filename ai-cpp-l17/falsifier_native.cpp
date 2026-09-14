#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <random>
#include <vector>

namespace nb = nanobind;

namespace
{
    [[nodiscard]] std::vector<int> make_shuffled(int n, std::uint32_t seed)
    {
        std::vector<int> v(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
        {
            v[static_cast<std::size_t>(i)] = i;
        }
        std::mt19937 rng(seed);
        std::shuffle(v.begin(), v.end(), rng);
        return v;
    }

    void insertion_sort(std::vector<int> &v)
    {
        for (std::size_t i = 1; i < v.size(); ++i)
        {
            int const key = v[i];
            std::size_t j = i;
            while (j > 0 && v[j - 1] > key)
            {
                v[j] = v[j - 1];
                --j;
            }
            v[j] = key;
        }
    }

    template <typename Sorter>
    [[nodiscard]] double median_nanoseconds(int n, int trials, Sorter sorter)
    {
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(trials));
        for (int t = 0; t < trials; ++t)
        {
            std::vector<int> v = make_shuffled(n, static_cast<std::uint32_t>(t));
            auto const start = std::chrono::steady_clock::now();
            sorter(v);
            auto const end = std::chrono::steady_clock::now();
            samples.push_back(std::chrono::duration<double, std::nano>(end - start).count());
        }
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2];
    }
} // namespace

// The claim under test: "std::sort is always faster than insertion sort."
// Returns the median wall-clock time, in nanoseconds, to sort a random
// permutation of `n` ints, `trials` independent runs.
[[nodiscard]] double time_std_sort(int n, int trials)
{
    return median_nanoseconds(n, trials, [](std::vector<int> &v)
                               { std::sort(v.begin(), v.end()); });
}

[[nodiscard]] double time_insertion_sort(int n, int trials)
{
    return median_nanoseconds(n, trials, [](std::vector<int> &v)
                               { insertion_sort(v); });
}

NB_MODULE(falsifier_native, m)
{
    m.def("time_std_sort", &time_std_sort, nb::arg("n"), nb::arg("trials"));
    m.def("time_insertion_sort", &time_insertion_sort, nb::arg("n"), nb::arg("trials"));
}
