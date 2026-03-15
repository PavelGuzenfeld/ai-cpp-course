# Section 4 Quiz: Nanobind Framework -- Zero-Copy C++ Bindings

## Q1: Why does accessing a Python `@property` repeatedly in a hot loop create a performance bottleneck?

- a) Python properties are computed using recursion
- b) Each property access goes through Python's descriptor protocol, involving function dispatch, argument parsing, and interpreter overhead
- c) Python properties are always stored on disk
- d) Properties in Python are thread-locked by default

**Answer: b)** Every `@property` access triggers Python's descriptor protocol: find the descriptor on the class, call its `__get__` method, parse arguments, and return the result. In a tracking loop processing hundreds of bounding boxes per frame at 30 FPS, these microseconds accumulate into milliseconds.

## Q2: What is the key difference between nanobind's `nb::ndarray` and pybind11's `py::array_t` for returning data to Python?

- a) `nb::ndarray` only supports integer arrays
- b) `nb::ndarray` can return a view backed by C++ memory with zero copies, while `py::array_t` typically copies the data
- c) `py::array_t` is faster because it uses GPU memory
- d) They are identical in functionality but `nb::ndarray` has a shorter name

**Answer: b)** `nb::ndarray` can wrap a raw C++ data pointer and expose it as a numpy array without any memcpy. This means returning a view into a C++ circular buffer or pre-allocated pool incurs zero allocation and zero copy overhead.

## Q3: How does nanobind's `def_rw` for struct attributes compare to Python's `@property` in terms of performance?

- a) They have identical performance
- b) `def_rw` is slower because it must cross the language boundary
- c) `def_rw` provides direct attribute access that bypasses the Python descriptor protocol, making it faster
- d) `def_rw` is only faster for string attributes

**Answer: c)** `def_rw` exposes a C++ struct field as a direct attribute, bypassing the descriptor protocol entirely. The access is a single memory read at a known offset, versus the multi-step protocol that `@property` requires.

## Q4: What practical problem does a C++ buffer pool with `nb::ndarray` returns solve?

- a) It prevents Python from running out of variable names
- b) It eliminates per-frame memory allocation by returning pre-allocated buffers as numpy arrays with zero overhead
- c) It compresses images automatically
- d) It converts all data to float64 for numerical stability

**Answer: b)** In a tracking loop, allocating and deallocating numpy arrays every frame wastes time on malloc/free and increases garbage collection pressure. A C++ pool allocates buffers once at startup and returns them as `nb::ndarray` views, reducing per-frame allocation to zero.

## Q5: When `HistoryNP.latest()` calls `.copy()` on every invocation, what waste does it introduce?

- a) It corrupts the original data
- b) It allocates a new array and copies all data on every call, even when the caller only needs a read-only view of the same frame
- c) It converts the data to a different type
- d) It moves the data from CPU to GPU

**Answer: b)** Calling `.copy()` allocates a fresh numpy array and copies the entire slice each time. If the caller uses the data immediately and does not need it to survive past the current frame, a zero-copy view (slice without `.copy()`) is sufficient and free.

## Q6: How does nanobind handle a C++ `std::invalid_argument` exception thrown during a bound function call?

- a) It terminates the Python interpreter
- b) It silently ignores the exception and returns None
- c) It automatically translates it into a Python `ValueError`
- d) It logs the error to a file and continues

**Answer: c)** Nanobind provides built-in exception translation: `std::invalid_argument` becomes `ValueError`, `std::out_of_range` becomes `IndexError`, `std::runtime_error` becomes `RuntimeError`, and so on.

## Q7: What does the `NB_STATIC` flag do when passed to `nanobind_add_module` in CMake?

- a) It prevents the module from being imported more than once
- b) It statically links the nanobind runtime into the module, producing a self-contained `.so` with no external nanobind dependencies
- c) It disables all dynamic memory allocation in the module
- d) It enables static type checking in Python

**Answer: b)** `NB_STATIC` embeds the nanobind runtime directly into your compiled module so it does not depend on a separate nanobind shared library at runtime. This simplifies deployment and avoids version conflicts.

## Q8: Nanobind produces binaries that are approximately what size compared to pybind11?

- a) About the same size
- b) 2x larger
- c) 3-5x smaller
- d) 10x smaller

**Answer: c)** Nanobind's clean-sheet redesign reduces template bloat and uses static dispatch instead of virtual dispatch for type casters, resulting in compiled modules that are 3-5x smaller than equivalent pybind11 modules.
