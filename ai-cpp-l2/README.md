# Lesson 2: Image Processing — Cache Locality and Execution Policies

## Goal

Crop and resize an image in C++ using OpenCV, then benchmark three execution
modes — scalar, SIMD (`unseq`), and parallel (`par`). Understand *why* cache
locality matters for image processing and how to measure it.

## The Memory Hierarchy

Every data access passes through a hierarchy of increasingly slower storage.
Understanding this hierarchy is the single most important concept for writing
fast C++ — and for understanding why Python/NumPy is slower than it should be.

| Level | Size | Latency | Bandwidth |
|-------|------|---------|-----------|
| L1 cache | 32-64 KB per core | **0.5 ns** (4 cycles) | ~500 GB/s |
| L2 cache | 256 KB-1 MB per core | **7 ns** (~14 cycles) | ~200 GB/s |
| L3 cache | 8-32 MB shared | **20 ns** (~40 cycles) | ~100 GB/s |
| RAM | 16-64 GB | **100 ns** (~200 cycles) | ~50 GB/s |
| SSD | TB | **100,000 ns** (100 µs) | ~3 GB/s |
| Network | - | **1,000,000+ ns** (1 ms) | variable |

### What This Means for Image Processing

A 1920x1080 RGB image is ~6 MB. It fits in L3 but not L1 or L2. If your code
accesses pixels in order (row by row), the hardware prefetcher loads the next
cache line before you need it. If your code accesses pixels randomly, every
access may be a cache miss — **200x slower** than a cache hit.

### Row-Major vs Column-Major

OpenCV and NumPy store images in **row-major** order (C-style):

```
Pixel (0,0) → Pixel (0,1) → Pixel (0,2) → ... → Pixel (0, W-1) →
Pixel (1,0) → Pixel (1,1) → ...
```

Processing row-by-row follows memory order (cache-friendly).
Processing column-by-column jumps across rows (cache-hostile).

## The C++ Code: `cpp_image_processor.cpp`

The implementation has three modes, all doing the same nearest-neighbor resize:

### Scalar (Sequential)

```cpp
for (int y = 0; y < target_height; ++y) {
    int src_y = static_cast<int>(y * y_ratio);
    process_row(cropped.ptr<uchar>(src_y),
                resized.ptr<uchar>(y),
                cropped.cols, x_ratio, target_width);
}
```

Plain `for` loop. One row at a time, one pixel at a time.

### SIMD (`std::execution::unseq`)

```cpp
std::for_each(std::execution::unseq, rows.begin(), rows.end(), [&](int y) {
    int src_y = static_cast<int>(y * y_ratio);
    process_row(cropped.ptr<uchar>(src_y),
                resized.ptr<uchar>(y),
                cropped.cols, x_ratio, target_width);
});
```

Same logic, but the compiler can vectorize pixel operations within each row
using SIMD instructions.

### Parallel (`std::execution::par`)

```cpp
std::for_each(std::execution::par, rows.begin(), rows.end(), [&](int y) {
    // ... same per-row logic ...
});
```

Each row is processed on a different CPU core. For a 1080-row image on an
8-core CPU, ~135 rows per core.

### The Pixel Copy Loop

```cpp
void process_row(const uchar* src_row, uchar* dst_row,
                 int src_cols, float x_ratio, int dst_cols) {
    for (int x = 0; x < dst_cols; ++x) {
        int src_x = static_cast<int>(x * x_ratio);
        dst_row[x * 3 + 0] = src_row[src_x * 3 + 0];  // B
        dst_row[x * 3 + 1] = src_row[src_x * 3 + 1];  // G
        dst_row[x * 3 + 2] = src_row[src_x * 3 + 2];  // R
    }
}
```

This is **cache-friendly**: reading `src_row` and writing `dst_row` both follow
sequential memory addresses.

## The Python Benchmark: `crop_resize.py`

Compares four approaches:
1. **Python/OpenCV**: `cv2.resize(cropped, ...)` — calls C internally but has
   Python overhead for the crop + function call
2. **C++ scalar**: Sequential nearest-neighbor
3. **C++ unseq**: SIMD-enabled nearest-neighbor
4. **C++ par**: Multi-threaded nearest-neighbor

## [OpenCV](https://opencv.org/) C++ Integration

OpenCV's `cv::Mat` is the C++ equivalent of a NumPy array for images:

```cpp
cv::Mat img(height, width, CV_8UC3, numpy_buffer_ptr);
cv::Rect roi(start_x, start_y, crop_w, crop_h);
cv::Mat cropped(img, roi);  // Zero-copy view into img
```

Key points:
- `CV_8UC3` = 8-bit unsigned, 3 channels (BGR)
- `cv::Mat` supports zero-copy views (like NumPy slicing)
- `ptr<uchar>(row)` returns a raw pointer to row data — no bounds checking,
  maximum speed

## [pybind11](https://github.com/pybind/pybind11) NumPy Integration

Converting between NumPy arrays and `cv::Mat`:

```cpp
py::array_t<uint8_t> crop_and_resize(py::array_t<uint8_t> input_image, ...) {
    py::buffer_info buf = input_image.request();
    cv::Mat img(buf.shape[0], buf.shape[1], CV_8UC3, buf.ptr);
    // ... process ...
    return py::array_t<uint8_t>(
        {result.rows, result.cols, result.channels()},
        {result.step[0], result.step[1], 1},
        result.data
    );
}
```

The strides `{step[0], step[1], 1}` tell NumPy how to navigate the memory
layout. This avoids any data transposition.

## In-Place Operations

Allocating a new array for every operation wastes time and memory:

```python
# Bad: 3 allocations
img = img.astype(float)           # alloc 1
img = img / 255.0                  # alloc 2
img = (img - mean) / std           # alloc 3

# Better: 1 allocation, in-place
img = img.astype(float)
img /= 255.0
img -= mean
img /= std
```

In C++, you control memory layout exactly. The crop-and-resize code allocates
the output buffer once and writes directly into it.

## `constexpr` — Compile-Time Computation

When a value is known at compile time, mark it `constexpr`:

```cpp
constexpr int CHANNELS = 3;
constexpr float INV_255 = 1.0f / 255.0f;
```

The compiler replaces these with literal constants — no runtime computation,
no memory load.

## Build and Run

```bash
# Inside Docker container
cd /workspace

# Build
colcon build --packages-select cpp_image_processor
source install/setup.bash

# Run benchmark (needs a BMP image in ai-cpp-l2/)
python3 ai-cpp-l2/crop_resize.py
```

## Exercises

1. **Reverse the loop order**: Process columns instead of rows in the scalar
   version. Measure the performance difference. Why is it slower?

2. **Measure with larger images**: Try 4K (3840x2160) and 8K (7680x4320)
   synthetic images. How does the `par` speedup scale?

3. **Add bilinear interpolation**: The current implementation uses nearest-
   neighbor. Implement bilinear as a new mode and benchmark it. Bilinear
   touches 4 source pixels per output pixel — how does this affect cache
   behavior?

4. **Profile cache misses**: Run the scalar version under [`perf stat`](https://perf.wiki.kernel.org/) `-e
   cache-misses` (inside the Docker container) and compare row-order vs
   column-order access.

5. **Compare with OpenCV's resize**: OpenCV internally uses highly optimized
   SIMD code. How close does the `par` version get?

## What You Learned

- The memory hierarchy determines performance more than the algorithm
- Row-major traversal is cache-friendly; column-major is not
- `std::execution::unseq` enables SIMD without writing intrinsics
- `std::execution::par` enables multi-threading without writing thread code
- OpenCV `cv::Mat` and NumPy share the same memory layout (row-major BGR)
- `constexpr` moves computation from runtime to compile time

## Lesson Files

| File | Description |
|------|-------------|
| [cpp_image_processor.cpp](cpp_image_processor.cpp) | C++ crop and resize with execution policies |
| [crop_resize.py](crop_resize.py) | Python benchmark comparing all approaches |
| [opencv_benchmark.py](opencv_benchmark.py) | OpenCV performance measurement script |
| [CMakeLists.txt](CMakeLists.txt) | CMake build configuration |
| [bmp-2048x1365.bmp](bmp-2048x1365.bmp) | Test image for benchmarking |
