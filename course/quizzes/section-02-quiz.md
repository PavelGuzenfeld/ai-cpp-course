# Section 2 Quiz: Image Processing -- Cache Locality and Execution Policies

## Q1: Why is processing an image row-by-row faster than column-by-column?

- a) Rows contain fewer pixels than columns
- b) Row-by-row traversal follows memory layout (row-major order), enabling hardware prefetching and cache line reuse
- c) The CPU has special row-processing instructions
- d) Column access requires more mathematical operations per pixel

**Answer: b)** Images are stored in row-major order, so consecutive pixels in a row occupy consecutive memory addresses. The hardware prefetcher loads the next cache line before you need it, making sequential row access nearly free compared to random column access that causes cache misses.

## Q2: An L1 cache miss that falls through to RAM is approximately how much slower than an L1 cache hit?

- a) 2x slower
- b) 10x slower
- c) 200x slower
- d) 1000x slower

**Answer: c)** L1 latency is approximately 0.5 ns while RAM latency is approximately 100 ns, making a RAM access about 200x slower than an L1 hit.

## Q3: What is the difference between `std::execution::unseq` and `std::execution::par`?

- a) `unseq` uses multiple threads; `par` uses SIMD
- b) `unseq` enables SIMD vectorization within a thread; `par` distributes work across multiple CPU cores
- c) They are identical in behavior but have different syntax
- d) `unseq` is faster for small data; `par` is faster for large data regardless of the operation

**Answer: b)** `unseq` tells the compiler to vectorize operations using SIMD instructions on a single core. `par` distributes iterations across multiple OS threads, allowing different cores to process different rows simultaneously.

## Q4: What does `constexpr` accomplish when used for a value like `CHANNELS = 3`?

- a) It makes the variable mutable at runtime
- b) It forces the compiler to evaluate the value at compile time, embedding it as a literal constant with no runtime memory load
- c) It allocates the variable on the GPU
- d) It makes the variable thread-safe

**Answer: b)** `constexpr` guarantees compile-time evaluation. The compiler replaces every use of the variable with its literal value, eliminating both the runtime computation and the memory load that a regular variable would require.

## Q5: A 1920x1080 RGB image is approximately 6 MB. Which cache level can hold it entirely?

- a) L1 cache (32-64 KB)
- b) L2 cache (256 KB - 1 MB)
- c) L3 cache (8-32 MB)
- d) None -- it must always be in RAM

**Answer: c)** At ~6 MB, the image exceeds L1 and L2 capacity but fits within a typical L3 cache (8-32 MB shared across cores).

## Q6: Why does `cv::Mat` support zero-copy views (like `cv::Mat cropped(img, roi)`)?

- a) It copies only the header metadata, while the pixel data is shared between the original and the view
- b) It compresses the cropped region to fit in less memory
- c) It uses GPU memory which is inherently shared
- d) It stores only the changed pixels

**Answer: a)** A `cv::Mat` created from a region of interest shares the underlying pixel buffer with the parent. Only the header (pointer, dimensions, stride) is new. This is analogous to NumPy slicing.

## Q7: In Python, which pattern reduces unnecessary memory allocations during image normalization?

- a) Creating a new array for each operation: `img = img / 255.0; img = img - mean`
- b) Using in-place operators: `img /= 255.0; img -= mean`
- c) Converting to a Python list first, then operating element-by-element
- d) Using string formatting to represent pixel values

**Answer: b)** In-place operators modify the existing array without allocating new memory. The non-in-place version creates a new temporary array for each operation, wasting both time and memory.

## Q8: For an 8-core CPU processing a 1080-row image with `std::execution::par`, approximately how many rows does each core handle?

- a) 1080
- b) 540
- c) 135
- d) 8

**Answer: c)** 1080 rows / 8 cores = 135 rows per core. The parallel execution policy distributes rows across available cores, with each core processing its share independently.
