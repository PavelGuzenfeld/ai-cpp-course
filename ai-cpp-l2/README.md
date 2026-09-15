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

## Stride Is Not Width

Everything above assumes row `n` starts at `n * width * bpp`. On a real
hardware buffer that is usually false — rows are padded so each one starts
at an alignment boundary the pixel format itself never mentions:

```
row_bytes = width * bpp                    stride = row_bytes + padding

row 0: [ pixel data ............ ][ pad ]  <- row 0 starts at offset 0
row 1: [ pixel data ............ ][ pad ]  <- row 1 starts at offset `stride`, not `row_bytes`
row 2: [ pixel data ............ ][ pad ]
```

NumPy hides this well enough that the assumption rarely gets tested: `arr`
carries its own `.strides` and every NumPy operation honors them, so code
that never has to compute a row offset by hand never has to get it right.
The moment you hand a pointer and a couple of ints to C++ — a `memcpy` loop,
a buffer handed across a C API, an NVMM/GPU surface — nothing enforces that
for you anymore.

`stride_view.hpp` defines a `View` that carries `stride` alongside `width`/
`height`/`channels`, and three copy functions built on it:

- `copy_using_stride` — reads each row at `src.stride`. Correct regardless
  of how much padding a row carries.
- `copy_using_width_as_stride` — assumes `stride == width * channels`. Row 0
  is unaffected (both formulas agree at offset 0); every row after it reads
  from the wrong offset. The output is the same size as a correct copy and
  contains real pixel bytes — just the wrong ones, shifted by a growing
  amount each row. **This is the failure that should worry you more than a
  crash**: nothing about the output says anything is wrong.
- `copy_overrunning_dst` — the write-side version of the same bug. The
  destination is sized `width * height * channels` (the no-padding
  assumption baked into an allocation instead of a read), but the copy loop
  advances the destination pointer by the source's real `stride` per row.
  Once `stride > width * channels`, the last row's write lands past the end
  of the allocation.

### Why the crash is nowhere near the bug

This course's own source material (`gst-nvmm-cpp`) shipped exactly this
mistake twice: a buffer pool stamped metadata using a stride recomputed from
the format descriptor instead of read off the real surface, and a `map()`
call handed back plane 0's pointer with `size` set to the *total* surface
size instead of one plane's — any full-size write ran off the end of plane
0. A third case sized a host copy as `width * 4 * height` against a surface
whose real pitch was larger; the result was a segfault in the next,
unrelated call, not the copy itself — the corrupted heap byte only became
visible once something else's allocation landed on it.

`stride_demo`'s `--trigger-overrun` flag reproduces the write-side version
of this deterministically. Built normally, the corruption is silent until
some later allocation trips over it (on this host, glibc's own heap
consistency check happens to catch it at the `delete`/`free` on the way
out — that is a coincidence of allocator internals, not a guarantee).
Built with `ENABLE_SANITIZERS=ON`, ASan reports the exact write:

```
==NNN==ERROR: AddressSanitizer: heap-buffer-overflow ...
WRITE of size 6144 at 0x... thread T0
    #1 ... in stride_lesson::copy_overrunning_dst(...) stride_view.hpp:87
    #2 ... in main stride_demo.cpp:109
```

That precision is the point of running a sanitizer *at the site of the
bug* instead of relying on whatever downstream symptom happens to surface
first — the stack trace above names the actual write; a stack trace from
the later, unrelated crash would not.

### Multi-plane buffers: why a plane pointer is not `base + w*h`

A YUV surface (NV12, I420) is not one buffer of interleaved samples — it is
several *planes*, each with its own stride, and the planes are not
necessarily contiguous with each other (hardware surfaces routinely pad
between planes too, for the same alignment reasons a single plane's rows
are padded). `plane[1] = base + width * height` is exactly as wrong as
`row(y) = base + y * width * bpp`, for the same reason: it recomputes an
offset the format spec implies instead of reading the one the surface
actually has. Treat a multi-plane handle as N independent `View`s, each
with its own base pointer and its own stride — never one buffer sliced by
arithmetic.

Run it:

```bash
cd ai-cpp-l2
../build/cpp_image_processor/stride_demo                    # correct + sheared
../build/cpp_image_processor/stride_demo --trigger-overrun   # + the overrun
```

### A version-skew bug found writing this lesson's own tests

`test_image_processing.py`'s tests initially failed with every output byte
equal to the *input's first byte*, repeated — for every call, regardless
of what the correct/sheared distinction should have produced. The C++
logic itself was correct: a standalone `g++` build of the identical source
file produced the right answer.

The difference was which `pybind11` the two builds saw. This image has two
installs — an apt package (2.9.1) at `/usr/include/pybind11`, and a pip
package (3.1.0, what `python3 -m pybind11 --includes` reports) — and
CMake's `find_package(pybind11)` resolves to the old apt one (no explicit
`-I` for pybind11 shows up in the actual compile command; it is found on
the default system include path). `py::array_t<std::uint8_t>(count)` binds
to a `(ssize_t count, const T *ptr = nullptr, ...)` constructor that exists
in both versions, but produces a differently-strided (and here, broken)
array under 2.9.1. The fix — construct with an explicit shape vector,
`py::array_t<std::uint8_t>(std::vector<py::ssize_t>{count})` — goes
through the unambiguous shape-based path on both versions.

The lesson this reinforces is the same one the section above teaches:
letting an API's convenient-looking short form silently produce a
different memory layout than intended is exactly the class of bug this
whole lesson is about, just one level up from a raw pointer and a stride.

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

## Cache Warming: The HFT Pattern That Usually Backfires

An HFT-style pattern you will see in trading engine code: *before* the signal
fires, pre-read the data the hot path will touch so L1/L2 already hold the
relevant lines when the trade decision arrives. The paper by Bilokon &
Gunduz (Imperial, 2023) reports a 90% win from this pattern.

At the micro-benchmark level, **cache warming is usually a net loss**. The
failure mode is that the warm-up itself touches far more memory than the hot
path it is trying to speed up.

Reproduce the effect with `cache_warming_fails.cpp`:

```bash
g++ -O2 -std=c++23 cache_warming_fails.cpp -o cache_warming_fails
./cache_warming_fails
```

Typical output on a recent x86-64:

```
cold:     91,252 ns/iter   (K=65,536 random reads)
warm:    655,255 ns/iter   (N=4,194,304 walk + K=65,536 random reads)
ratio warm/cold = 7.18x   (warming is a NET LOSS here)
```

Warming a 16 MiB array to serve 64 KiB of random reads does **64× more
memory traffic** than the hot path it is priming. No cache strategy survives
that imbalance.

### When cache warming actually pays

Two conditions, both required:

1. **The warm-up is amortised across an outer loop** — a trading engine walks
   its order book on every market tick, so by the time a real signal fires,
   the caches are hot as a side effect. The cost is spread across the entire
   tick stream, not charged to each trade decision.
2. **The hot path accesses significantly more memory than the warm-up** — the
   formula you want is *warm_up_bytes << hot_path_bytes*. In
   `cache_warming_fails.cpp` the ratio is backwards, which is why the pattern
   loses.

**The rule to remember:** *cache warming is a system-design pattern, not a
function-level optimisation.* If you see it in a PR description and the
warm-up pass is larger than the measured pass, flag it for review.

---

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
| [cache_warming_fails.cpp](cache_warming_fails.cpp) | Standalone demo of the cache-warming anti-pattern |
| [stride_view.hpp](stride_view.hpp) | Stride-aware `View` type; correct, sheared, and overrunning copy functions |
| [stride_demo.cpp](stride_demo.cpp) | Standalone demo: correct vs sheared vs (optionally) overrunning copy on the padded BMP |
| [stride_view_native.cpp](stride_view_native.cpp) | pybind11 bindings for the correct/sheared copies, for unit testing |
| [test_image_processing.py](test_image_processing.py) | Unit tests: hand-computed padded buffers, exact shear byte values |
| [test_integration_image_processing.py](test_integration_image_processing.py) | Integration test: the real BMP asset, padded, copied both ways |
| [CMakeLists.txt](CMakeLists.txt) | CMake build configuration |
| [bmp-2048x1365.bmp](bmp-2048x1365.bmp) | Test image for benchmarking |
