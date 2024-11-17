#include <algorithm>
#include <execution>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

std::vector<int> portable_simd_sum_vectors(const std::vector<int> &a, const std::vector<int> &b)
{
    if (a.size() != b.size())
    {
        throw std::runtime_error("Vectors must be the same size");
    }
    std::vector<int> result(a.size());

    // parallel and vectorized execution
    std::transform(std::execution::unseq, a.begin(), a.end(), b.begin(), result.begin(), [](int x, int y)
                   { return x + y; });

    return result;
}

PYBIND11_MODULE(portable_simd_sum_vectors, m)
{
    m.def("portable_simd_sum_vectors", &portable_simd_sum_vectors, "Portable SIMD sum of two vectors");
}
