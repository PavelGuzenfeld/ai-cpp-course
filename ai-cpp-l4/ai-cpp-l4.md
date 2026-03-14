# Lesson 4: Nanobind Framework — Zero-Copy C++ Bindings for Python

## Why Nanobind Over Pybind11?

In Lessons 1-3 we used pybind11 to wrap C++ code for Python. Pybind11 works, but it
carries significant overhead that matters when you're optimizing hot paths:

| Metric | pybind11 | nanobind |
|--------|----------|----------|
| Binary size | ~150 KB per module | ~30-50 KB per module (3-5x smaller) |
| Compile time | Slow (heavy template instantiation) | 2-3x faster |
| numpy interop | `py::array_t<T>` with copies | `nb::ndarray` with zero-copy views |
| Type caster overhead | Virtual dispatch | Static dispatch |

Nanobind was created by Wenzel Jakob (the original author of pybind11) as a clean-sheet
redesign. It targets Python 3.8+ and C++17+, dropping legacy compatibility in exchange
for smaller, faster bindings.

### Key advantages for CV/tracking workloads:

1. **Zero-copy `ndarray`**: Return numpy arrays backed by C++ memory — no memcpy.
2. **Smaller binaries**: Less code pulled into each `.so`, faster import time.
3. **Faster compile**: Less template bloat means faster iteration.
4. **Better move semantics**: First-class support for move-only types.

## Motivation: tracker_engine Bottlenecks

The [tracker_engine](https://github.com/thebandofficial/tracker_engine) codebase has
several hot-path bottlenecks that nanobind can eliminate:

### 1. BoundingBox — Death by @property

```python
class BoundingBox:
    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def area(self):
        return self.w * self.h
```

Every `@property` access goes through Python's descriptor protocol. In a tracking loop
processing hundreds of boxes per frame at 30 FPS, this adds up to milliseconds of pure
Python overhead. A C++ struct with nanobind-exposed properties eliminates this entirely.

### 2. NumpyBufferPool — Python in the Fast Path

The pool's `acquire()` and `release()` methods are pure Python. Even with pre-allocated
numpy arrays, each call pays for Python function dispatch, argument parsing, and GIL
operations. A C++ pool with `nb::ndarray` returns can serve buffers with near-zero
overhead.

### 3. HistoryNP.latest() — Copies Instead of Views

```python
def latest(self, n=1):
    return self.buffer[:n].copy()  # memcpy every time!
```

With `nb::ndarray`, we can return a *view* into C++ memory — no copy, no allocation.
The caller sees a numpy array that directly references the circular buffer's storage.

## nanobind::ndarray — Zero-Copy Numpy Interop

The `nb::ndarray` type is nanobind's killer feature for numerical code:

```cpp
#include <nanobind/ndarray.h>

// Accept any numpy array of doubles
void process(nb::ndarray<double> input);

// Return a view into C++ memory as a numpy array
nb::ndarray<nb::numpy, double, nb::shape<4>> to_array() {
    return nb::ndarray<nb::numpy, double, nb::shape<4>>(
        data_ptr, {4}, nb::handle()
    );
}
```

Key points:
- **No copy on input**: nanobind accesses the numpy array's data pointer directly.
- **No copy on output**: You can return arrays backed by C++ memory.
- **Shape/dtype checking**: Template parameters enforce shape and type at the binding layer.
- **Framework-agnostic**: Works with numpy, PyTorch, JAX, etc.

## Binding C++ Classes with Properties

```cpp
nb::class_<BBox>(m, "BBox")
    .def(nb::init<double, double, double, double>())
    .def_rw("x", &BBox::x)           // read-write attribute
    .def_prop_ro("area", &BBox::area) // read-only computed property
    .def("iou", &BBox::iou);          // method
```

- `def_rw` / `def_ro`: Direct attribute access (no Python descriptor overhead).
- `def_prop_ro` / `def_prop_rw`: Computed properties — still C++ speed.
- Properties in nanobind are faster than pybind11 because the type caster uses
  static dispatch instead of virtual dispatch.

## Exception Translation

Nanobind automatically translates C++ exceptions to Python:

| C++ Exception | Python Exception |
|---------------|-----------------|
| `std::invalid_argument` | `ValueError` |
| `std::out_of_range` | `IndexError` |
| `std::runtime_error` | `RuntimeError` |
| `std::bad_alloc` | `MemoryError` |

You can also register custom translators:

```cpp
nb::register_exception_translator([](const std::exception_ptr &p, void *) {
    try { std::rethrow_exception(p); }
    catch (const MyError &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
});
```

## Building with CMake

Nanobind provides a `nanobind_add_module` CMake function:

```cmake
find_package(nanobind CONFIG REQUIRED HINTS /usr/local/nanobind/cmake)
nanobind_add_module(my_module NB_STATIC my_module.cpp)
```

The `NB_STATIC` flag statically links the nanobind runtime, producing a self-contained
`.so` with no external dependencies beyond Python.

## Build and Run

```bash
# Inside Docker container
cd /workspace
colcon build --packages-select nanobind-l4
source install/setup.bash

# Run benchmarks
python3 ai-cpp-l4/benchmark_nanobind.py

# Run tests
pytest ai-cpp-l4/ -v
```

## What You Learned

- nanobind produces 3-5x smaller binaries and compiles 2-3x faster than pybind11
- `nb::ndarray` enables zero-copy numpy interop — no memcpy for input or output
- C++ struct properties via `def_rw`/`def_prop_ro` are faster than Python `@property`
- Pre-allocated buffer pools eliminate per-frame allocation overhead
- Circular buffers can return views instead of copies when data is contiguous
- C++ exceptions automatically map to Python exceptions (ValueError, IndexError, etc.)

## Exercises

1. **BoundingBox speedup**: Run `benchmark_nanobind.py` and compare Python vs C++
   BoundingBox property access. How many times faster is the C++ version?

2. **Buffer pool zero-copy**: Modify the benchmark to verify that `acquire()` returns
   memory that is *not* re-allocated between acquire/release cycles (hint: check
   `ndarray.data` pointer values).

3. **History view semantics**: Write a test that modifies the underlying circular buffer
   after calling `latest()` and verify that the returned view reflects the change
   (proving it is a view, not a copy).

4. **Add a new binding**: Add a `scale(factor)` method to BoundingBox that multiplies
   w and h by a factor. Expose it through nanobind and benchmark it against the
   Python equivalent.

5. **ndarray with shape constraints**: Modify `from_array()` to accept only arrays of
   exactly 4 elements. What happens when you pass an array of 5?

## Files in This Lesson

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Build configuration for all three nanobind modules |
| `bbox_native.cpp` | C++ BoundingBox with nanobind bindings |
| `bbox_slow.py` | Pure Python BoundingBox (the "before" version) |
| `buffer_pool_native.cpp` | Zero-overhead buffer pool with ndarray |
| `history_view_native.cpp` | Zero-copy circular buffer with view semantics |
| `benchmark_nanobind.py` | Performance comparison: Python vs C++ |
| `test_nanobind.py` | Unit tests for all three modules |
| `test_integration_nanobind.py` | Integration tests simulating real workloads |
