#include <algorithm> // for std::transform
#include <execution> // for std::execution::unseq
#include <pybind11/pybind11.h> // Pybind11 library
#include <vector> // for std::vector

auto portable_simd_sum_vectors (const std::vector<int> &a, const std::vector<int> &b) -> std::vector<int>
{
    // Assure that the vectors are the same size
    if (a.size() != b.size())
    {
        // throw an exception
        throw std::runtime_error("Vectors must be the same size");
    }
    
    
    // Create a vector to store the result with the same size as the input vectors
    std::vector<int> result(a.size());

    
    /*
        param1: Iterator to the first element of the first vector
        param2: Iterator to the last element of the first vector
        param3: Iterator to the first element of the second vector
        param4: Iterator to the first element of the result vector
        param5: Lambda function that takes two integers and returns their sum
        std::execution::unseq allows the compiler to use SIMD instructions
    */
    std::transform(std::execution::unseq, a.begin(), a.end(), b.begin(), result.begin(), 
    [](int x, int y)
                   { return x + y; });

    return result;
}

// module name: portable_simd_sum_vectors, as m
PYBIND11_MODULE(portable_simd_sum_vectors, m) 
{
    // module docstring
    m.doc() = "Portable SIMD sum of two vectors";
    // function docstring
    // function name, function pointer, function docstring
    m.def("portable_simd_sum_vectors", &portable_simd_sum_vectors, "Portable SIMD sum of two vectors");
}


// More details:
/*
    std::vector - a dynamic(on the heap) array that can change its size
    std::vector::size - returns the number of elements in a vector
    https://en.cppreference.com/w/cpp/container/vector

    std::transform - applies a function to each element of two ranges and stores the result in a third range
    https://en.cppreference.com/w/cpp/algorithm/transform

    std::execution::unseq - allows the compiler to use SIMD instructions
    Requires C++17 or later
    Linking with -ltbb is required
    https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag


    std::runtime_error - an exception that can be thrown when a runtime error occurs
    https://en.cppreference.com/w/cpp/error/runtime_error

    std::begin - returns an iterator to the beginning of a container
    std::end - returns an iterator to the end of a container
    https://en.cppreference.com/w/cpp/iterator/begin
    https://en.cppreference.com/w/cpp/iterator/end

*/
