# Section 4: Nanobind Framework -- Zero-Copy C++ Bindings for Python

## Video 4.1: Why Nanobind Over Pybind11 (~8 min)

### Slides
- Slide 1: The cost of pybind11 -- Binary size ~150 KB per module, slow compile times (heavy template instantiation), `py::array_t<T>` requires copies, virtual dispatch in type casters.
- Slide 2: Nanobind improvements -- Binary size ~30-50 KB (3-5x smaller), 2-3x faster compile, `nb::ndarray` with zero-copy views, static dispatch. Clean-sheet redesign targeting Python 3.8+ and C++17+.
- Slide 3: tracker_engine bottlenecks that nanobind fixes -- BoundingBox `@property` overhead (Python descriptor protocol on hundreds of boxes per frame at 30 FPS), NumpyBufferPool Python function dispatch, HistoryNP.latest() copies instead of views.
- Slide 4: Jetson note -- On Jetson, smaller binary sizes from nanobind matter more because storage and memory are constrained. Faster compile times also help since Jetson CPUs are slower than desktop for compilation.

### Key Takeaway
- nanobind produces smaller, faster bindings than pybind11 -- the zero-copy ndarray support eliminates the biggest overhead when passing numerical data between Python and C++.

## Video 4.2: nb::ndarray -- Zero-Copy Numpy Interop (~12 min)

### Slides
- Slide 1: The killer feature -- `nb::ndarray` accepts any numpy array of a given type without copying. On input, nanobind accesses the numpy array's data pointer directly. On output, you can return arrays backed by C++ memory.
- Slide 2: Input zero-copy -- `void process(nb::ndarray<double> input)` receives a pointer to numpy's data. No memcpy, no allocation. Shape and dtype checking via template parameters.
- Slide 3: Output zero-copy -- Return `nb::ndarray<nb::numpy, double, nb::shape<4>>` backed by C++ memory. The caller sees a numpy array that directly references C++ storage.
- Slide 4: Framework-agnostic -- ndarray works with numpy, PyTorch, JAX, and any framework supporting the DLPack protocol. Write bindings once, use with any framework.
- Slide 5: BoundingBox example -- C++ `BBox` struct with `def_rw("x", &BBox::x)` for direct attribute access (no Python descriptor overhead) and `def_prop_ro("area", &BBox::area)` for computed properties at C++ speed.

### Live Demo
- Run `benchmark_nanobind.py` showing Python BoundingBox vs C++ BoundingBox property access speed. Show the pointer values to prove zero-copy is working.

### Key Takeaway
- `nb::ndarray` enables true zero-copy data exchange between Python and C++ -- no memcpy for input or output, which eliminates the biggest overhead in numerical pipelines.

## Video 4.3: Buffer Pools and Circular Buffers (~10 min)

### Slides
- Slide 1: The buffer pool problem -- tracker_engine's `NumpyBufferPool.acquire()` and `release()` are pure Python. Even with pre-allocated numpy arrays, each call pays for Python function dispatch and GIL operations.
- Slide 2: C++ buffer pool -- Pre-allocate a fixed number of numpy arrays in C++. `acquire()` returns an ndarray backed by C++ memory. `release()` returns it to the pool. Near-zero overhead.
- Slide 3: The circular buffer problem -- tracker_engine's `HistoryNP.latest()` calls `.copy()` every time. With ndarray, we return a view into C++ memory -- no copy, no allocation.
- Slide 4: View safety -- Views are safe when the caller uses them immediately (same frame). For cross-frame storage, return a copy. The C++ buffer pool manages lifetime automatically.
- Slide 5: Exception translation -- nanobind automatically maps `std::invalid_argument` to `ValueError`, `std::out_of_range` to `IndexError`, `std::runtime_error` to `RuntimeError`. Custom translators are also supported.

### Key Takeaway
- Pre-allocated buffer pools and zero-copy views eliminate per-frame allocation overhead -- the two biggest performance drains in Python tracking loops.

## Video 4.4: Building and Testing Nanobind Modules (~8 min)

### Slides
- Slide 1: CMake integration -- `find_package(nanobind CONFIG REQUIRED)`, `nanobind_add_module(my_module NB_STATIC my_module.cpp)`. The `NB_STATIC` flag statically links the nanobind runtime for a self-contained .so.
- Slide 2: Building with colcon -- `colcon build --packages-select nanobind-l4`, `source install/setup.bash`, `pytest ai-cpp-l4/ -v`.
- Slide 3: Exercise overview -- BoundingBox speedup measurement, buffer pool zero-copy verification (check ndarray.data pointer values), history view semantics (modify buffer, verify view reflects change), add new bindings.

### Live Demo
- Build the nanobind module, run the full test suite, show the benchmark results comparing Python vs C++ implementations for BBox, BufferPool, and HistoryBuffer.

### Key Takeaway
- nanobind modules build cleanly with CMake and integrate seamlessly into Python test workflows -- the same pytest commands work for testing both Python and C++ implementations.
