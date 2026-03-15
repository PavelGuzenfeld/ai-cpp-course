# Section 2: Image Processing -- Cache Locality and Execution Policies

## Video 2.1: The Memory Hierarchy (~10 min)

### Slides
- Slide 1: The memory hierarchy table -- L1 cache (32-64 KB, 0.5 ns), L2 (256 KB-1 MB, 7 ns), L3 (8-32 MB, 20 ns), RAM (16-64 GB, 100 ns), SSD (TB, 100,000 ns), Network (1,000,000+ ns). Each level is 10-1000x slower than the previous.
- Slide 2: What this means for images -- A 1920x1080 RGB image is ~6 MB. It fits in L3 but not L1 or L2. Sequential access lets the hardware prefetcher load the next cache line before you need it. Random access means every access may be a cache miss (200x slower).
- Slide 3: Row-major vs column-major -- OpenCV and NumPy store images in row-major order. Processing row-by-row follows memory order (cache-friendly). Processing column-by-column jumps across rows (cache-hostile).
- Slide 4: The cache line -- CPUs fetch memory in 64-byte cache lines. Reading one byte loads the entire line. Sequential access means the next 63 bytes are already cached. Random access wastes the prefetch.
- Slide 5: Jetson cache note -- Jetson Orin has smaller caches than desktop CPUs (64 KB L1, 2 MB L2, 4 MB L3 shared). Cache efficiency matters even more on embedded platforms. The same code can be 2-3x slower on Jetson if it thrashes the cache.

### Key Takeaway
- The memory hierarchy determines performance more than the algorithm -- cache-friendly access patterns are the most important optimization.

## Video 2.2: C++ Image Processing with OpenCV (~12 min)

### Slides
- Slide 1: OpenCV cv::Mat basics -- The C++ equivalent of a NumPy array for images. `CV_8UC3` means 8-bit unsigned, 3 channels (BGR). Zero-copy views via `cv::Mat(img, roi)` just like NumPy slicing.
- Slide 2: The crop-and-resize implementation -- Three execution modes (scalar, SIMD, parallel) all doing nearest-neighbor resize. The pixel copy loop processes one row at a time for cache-friendly access.
- Slide 3: Scalar mode -- Plain `for` loop. One row at a time, one pixel at a time. Baseline for comparison.
- Slide 4: SIMD mode (`std::execution::unseq`) -- Same logic wrapped in `std::for_each` with `unseq` policy. Compiler can vectorize pixel operations within each row.
- Slide 5: Parallel mode (`std::execution::par`) -- Each row processed on a different CPU core. For 1080 rows on 8 cores, approximately 135 rows per core. Rows are independent so parallelism is safe.
- Slide 6: pybind11 NumPy integration -- Converting between `py::array_t<uint8_t>` and `cv::Mat`. The `buffer_info` struct gives access to shape, strides, and data pointer. Strides `{step[0], step[1], 1}` tell NumPy how to navigate the memory layout.

### Live Demo
- Walk through `cpp_image_processor.cpp`, showing the three execution modes. Run `crop_resize.py` and compare Python/OpenCV vs C++ scalar vs SIMD vs parallel.

### Key Takeaway
- `std::execution::par` enables multi-threaded image processing without writing thread code -- each row runs on a separate core.

## Video 2.3: constexpr and In-Place Operations (~8 min)

### Slides
- Slide 1: `constexpr` values -- When a value is known at compile time, mark it `constexpr`. The compiler replaces it with a literal constant -- no runtime computation, no memory load. Example: `constexpr int CHANNELS = 3;`.
- Slide 2: In-place operations matter -- Allocating a new array for every operation wastes time and memory. Comparison: 3 allocations vs 1 allocation with in-place `/=` and `-=` operators.
- Slide 3: C++ memory control -- In C++, you control memory layout exactly. The crop-and-resize code allocates the output buffer once and writes directly into it. No intermediate copies.
- Slide 4: Summary of what you learned -- Memory hierarchy determines performance, row-major traversal is cache-friendly, three execution policies (seq, unseq, par), OpenCV Mat and NumPy share the same memory layout, constexpr moves computation to compile time.

### Key Takeaway
- Controlling memory layout and avoiding unnecessary allocations is the foundation of high-performance C++ -- habits that pay off in every subsequent lesson.

## Video 2.4: Exercises and Cache Profiling (~8 min)

### Slides
- Slide 1: Exercise overview -- Reverse loop order (columns instead of rows), test with larger images (4K, 8K), add bilinear interpolation, profile cache misses with `perf stat`.
- Slide 2: Expected results from reversing loop order -- Column-major access on a 1920-wide image means each pixel access jumps 5760 bytes (1920 * 3 channels). This defeats the prefetcher and can be 5-10x slower.
- Slide 3: Using perf stat -- `perf stat -e cache-misses ./program` shows hardware cache miss counts. Comparing row-order vs column-order reveals the cost of cache-hostile access patterns.
- Slide 4: Jetson profiling note -- Jetson does not support `perf stat` in the same way as desktop Linux. Use NVIDIA Nsight Systems (`nsys`) instead for CPU and GPU profiling. The principles of cache-friendly access apply equally on ARM.

### Key Takeaway
- Measure cache behavior with hardware counters to understand why code is slow at the hardware level, not just the algorithmic level.
