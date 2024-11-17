import timeit
import numpy as np
import portable_simd_sum_vectors

# define pure python sum
def python_sum(a, b):
    return [x + y for x, y in zip(a, b)]

# define numpy sum
def numpy_sum(a, b):
    return np.add(a, b)

# setup
size = 10**6
a = list(range(size))
b = list(range(size))

# benchmarks
python_time = timeit.timeit(lambda: python_sum(a, b), number=10)
numpy_time = timeit.timeit(lambda: numpy_sum(np.array(a), np.array(b)), number=10)
cpp_simd_time = timeit.timeit(lambda: portable_simd_sum_vectors.portable_simd_sum_vectors(a, b), number=10)

# results
print(f"Python Sum: {python_time:.4f} seconds")
print(f"NumPy Sum: {numpy_time:.4f} seconds")
print(f"C++ SIMD Sum: {cpp_simd_time:.4f} seconds")
