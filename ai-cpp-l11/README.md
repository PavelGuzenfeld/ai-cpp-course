# Lesson 11: Memory Safety Without Sacrifice

## Goal

Write safe C++ that catches bugs at compile time — no segfaults, no leaks, no
undefined behavior — while keeping the zero-overhead performance that brought
you to C++ in the first place.

## Build and Run

Inside the Docker container:

```bash
cd /workspace

# Build with colcon
colcon build --packages-select nanobind-l11
source install/setup.bash

# Run the demo
python3 ai-cpp-l11/safety_demo.py

# Run unit tests
python3 -m pytest ai-cpp-l11/test_safety.py -v

# Run integration tests
python3 -m pytest ai-cpp-l11/test_integration_safety.py -v
```

Or build with CMake directly:

```bash
cd ai-cpp-l11
mkdir -p build && cd build
cmake .. -DCMAKE_CXX_COMPILER=g++-13
make -j$(nproc)

cd ..
PYTHONPATH=build python3 safety_demo.py
```

To run the ASAN example (standalone, not a Python module):

```bash
cd ai-cpp-l11/build
cmake .. -DCMAKE_CXX_COMPILER=g++-13 -DENABLE_SANITIZERS=ON
make asan_example -j$(nproc)
./asan_example        # Will print ASAN error reports for the buggy functions
./asan_example fixed  # Runs only the fixed versions (clean output)
```

## Why Memory Safety Matters

In `tracker_engine`, you never think about memory. Python handles allocation,
deallocation, null checks, and bounds checking automatically. There are no
segfaults. There are no double-frees. There are no buffer overflows.

The cost is performance. Every Python object is heap-allocated, reference-counted,
and type-checked at runtime. That overhead is why you are learning C++.

But raw C++ is dangerous. A single out-of-bounds write can corrupt memory
silently, crash hours later, or — worst of all — appear to work fine in testing
and explode in production. This lesson shows you how modern C++ eliminates
entire categories of bugs at compile time, with zero runtime cost.

The key insight: **safety and performance are not opposites.** The safest C++
patterns are often the fastest, because they give the compiler more information
to optimize.

---

## std::span — Safe Views Without Copies

### Python Equivalent

```python
# numpy slicing creates a view, not a copy
row = image[y, :]          # zero-cost view of one row
chunk = buffer[start:end]  # zero-cost slice
```

### The Problem: Raw Pointer + Size

```cpp
// Dangerous: nothing prevents you from reading past the end
void process_row(const double* data, int size) {
    for (int i = 0; i <= size; i++) {  // off-by-one: reads past end
        total += data[i];               // undefined behavior, no error
    }
}
```

The function signature does not encode the relationship between `data` and
`size`. The caller can pass the wrong size. The callee can ignore the size.
Nothing stops you.

### The Solution: std::span

```cpp
#include <span>

// Safe: span knows its own size
void process_row(std::span<const double> data) {
    for (double val : data) {   // range-for: cannot go out of bounds
        total += val;
    }
}
```

`std::span` is a non-owning view of contiguous memory. It carries a pointer
and a size together, just like a Python slice. It adds zero overhead in release
builds — it is literally the same two values you would pass manually, but now
the compiler can reason about them.

### Debug vs Release

```cpp
// In debug mode: throws std::out_of_range
double val = data[999];  // bounds-checked

// In release mode (-O3): identical to raw pointer access
double val = data[999];  // zero overhead, same assembly
```

### Slicing Is Free

```cpp
std::span<const double> row = buffer.subspan(y * width, width);
// No copy. No allocation. Just pointer arithmetic.
```

This is the C++ equivalent of `buffer[y*width : y*width + width]` in Python.

### What We Build: `safe_views.cpp`

The module provides:
- `span_sum()` — sums elements using std::span (safe, bounds-aware)
- `span_slice()` — returns a sub-span (zero-cost view)
- `safe_at()` — explicit bounds checking that raises IndexError on failure
- `raw_pointer_sum()` — raw pointer version for comparison

In release mode, `span_sum` and `raw_pointer_sum` produce identical assembly.
The safety is free.

---

## std::optional — No More Null Pointers

### Python Equivalent

```python
from typing import Optional

def find_target(frame) -> Optional[BBox]:
    """Returns None if no target is found."""
    detections = detect(frame)
    if not detections:
        return None
    return detections[0]

# Caller must handle None
result = find_target(frame)
if result is not None:
    track(result)
```

In `tracker_engine`, you use `hasattr()` and `None` checks to handle optional
components — an optional velocity estimator, an optional re-identification
module, an optional visualization overlay.

### The Problem: Null Pointers

```cpp
// Dangerous: nothing forces the caller to check for null
BBox* find_target(const Frame& frame) {
    if (no_detection)
        return nullptr;      // caller might forget to check
    return new BBox{...};    // who deletes this?
}

// Crash:
BBox* result = find_target(frame);
double area = result->area();  // segfault if null
```

### The Solution: std::optional

```cpp
#include <optional>

std::optional<BBox> find_target(const Frame& frame) {
    if (no_detection)
        return std::nullopt;  // explicit "no value"
    return BBox{10, 20, 100, 80};
}

// Caller is forced to check
auto result = find_target(frame);
if (result.has_value()) {
    double area = result->area();  // safe
}

// Or use value_or for a default
BBox box = result.value_or(BBox{0, 0, 0, 0});
```

`std::optional<T>` stores the value inline (no heap allocation) and a boolean
flag. The type system forces you to acknowledge that the value might not exist.
No null pointer. No segfault. No heap allocation.

### What We Build: `memory_safety_demo.cpp`

`OptionalDetection` wraps `std::optional<BBox>` and exposes:
- `has_value()` — check if a detection exists
- `value()` — get the detection (raises ValueError if empty)
- `value_or(default)` — get the detection or a default

---

## RAII — Resources Clean Themselves

### Python Equivalent

```python
# Context manager ensures cleanup
with open("model.bin", "rb") as f:
    data = f.read()
# f is closed here, even if an exception was thrown

with torch.cuda.device(0):
    tensor = torch.zeros(1000)
# GPU context is released here
```

### The Problem: Manual Resource Management

```cpp
void process() {
    double* buffer = new double[1000000];
    FILE* f = fopen("data.bin", "rb");

    fread(buffer, sizeof(double), 1000000, f);
    process_data(buffer);  // What if this throws?

    fclose(f);       // Never reached if process_data throws
    delete[] buffer; // Leaked!
}
```

If `process_data` throws an exception, `fclose` and `delete[]` are never
called. The file handle leaks. The memory leaks. In a long-running tracker,
these leaks accumulate and eventually crash the system.

### The Solution: RAII (Resource Acquisition Is Initialization)

```cpp
class Buffer {
    std::unique_ptr<double[]> data_;
    size_t size_;
public:
    Buffer(size_t n) : data_(std::make_unique<double[]>(n)), size_(n) {}
    // Destructor runs automatically when Buffer goes out of scope
    // No manual cleanup needed. Exception-safe. Cannot leak.

    std::span<double> view() { return {data_.get(), size_}; }
};

void process() {
    Buffer buf(1000000);  // allocated
    // ... use buf ...
}   // buf destroyed here — memory freed automatically
```

The rule is simple: **acquisition happens in the constructor, release happens in
the destructor.** Since C++ guarantees destructors run when objects leave scope
(even during exceptions), resources cannot leak.

### What We Build: `memory_safety_demo.cpp`

`RAIIBuffer` demonstrates:
- Constructor allocates memory
- Destructor frees memory (automatically, always)
- `size()` and `get(index)` for access
- Cannot leak, even if exceptions occur

---

## std::unique_ptr / std::shared_ptr — Ownership Is Clear

### Python Equivalent

```python
# Python uses reference counting — ownership is invisible
model = load_model("yolov8.pt")
tracker.set_model(model)  # Does tracker own it? Does the caller?
# Who cares — Python ref-counts it. Deleted when nobody references it.
```

Python's approach works but has costs: every object carries a reference count
(8 bytes), every assignment is an atomic increment/decrement, and cycles require
a garbage collector.

### The Problem: Who Deletes This?

```cpp
Model* load_model(const std::string& path) {
    return new Model(path);  // Caller must delete. But will they?
}

void setup() {
    Model* m = load_model("yolov8.onnx");
    tracker.set_model(m);    // Does tracker delete it?
    pipeline.set_model(m);   // Now two owners — who deletes?
}
// If nobody deletes: memory leak
// If both delete: double-free (crash or corruption)
```

### The Solution: Smart Pointers

**`std::unique_ptr` — exactly one owner:**

```cpp
auto model = std::make_unique<Model>("yolov8.onnx");
tracker.set_model(std::move(model));  // ownership transferred
// model is now nullptr — cannot accidentally use it
```

**`std::shared_ptr` — multiple owners, reference counted:**

```cpp
auto buffer = std::make_shared<Buffer>(1024);
tracker.set_buffer(buffer);    // ref count: 2
pipeline.set_buffer(buffer);   // ref count: 3
// Deleted when last shared_ptr goes out of scope
```

The ownership semantics are encoded in the type:
- `unique_ptr<T>` — "I own this, nobody else does"
- `shared_ptr<T>` — "Multiple owners, last one cleans up"
- `T&` — "I am borrowing this, someone else owns it"
- `T*` — "Non-owning pointer (observer), do not delete"

### What We Build: `memory_safety_demo.cpp`

- `UniqueModel` — wraps `std::unique_ptr<Model>`, demonstrates ownership transfer
- `SharedBuffer` — wraps `std::shared_ptr<Buffer>`, demonstrates reference counting

---

## [ASAN](https://clang.llvm.org/docs/AddressSanitizer.html) and [UBSAN](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html) — Catch Bugs Automatically

### What They Are

AddressSanitizer (ASAN) and UndefinedBehaviorSanitizer (UBSAN) are compiler
tools that instrument your code to detect bugs at runtime. They are your
safety net during development.

**ASAN catches:**
- Buffer overflow (read/write past array bounds)
- Use-after-free (accessing freed memory)
- Memory leaks (allocated but never freed)
- Double-free (freeing the same memory twice)
- Stack buffer overflow

**UBSAN catches:**
- Signed integer overflow
- Null pointer dereference
- Misaligned memory access
- Shift by negative or too-large amount

### How to Enable in CMake

```cmake
# Add to CMakeLists.txt
option(ENABLE_SANITIZERS "Enable ASAN and UBSAN" OFF)

if(ENABLE_SANITIZERS)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()
```

Build with sanitizers:

```bash
cmake .. -DENABLE_SANITIZERS=ON -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

### How to Read the Output

ASAN produces detailed reports:

```
=================================================================
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x...
WRITE of size 4 at 0x... thread T0
    #0 0x... in buffer_overflow() asan_example.cpp:8
    #1 0x... in main asan_example.cpp:25

0x... is located 0 bytes to the right of 40-byte region [0x..., 0x...)
allocated by thread T0 here:
    #0 0x... in operator new[](unsigned long)
    #1 0x... in buffer_overflow() asan_example.cpp:5
```

The report tells you:
1. What happened (heap-buffer-overflow)
2. Where it happened (file:line)
3. Where the memory was allocated
4. The exact byte offset of the violation

### What We Build: `asan_example.cpp`

A standalone program (not a Python module) with intentionally buggy functions:
- `buffer_overflow()` — writes past the end of an array
- `use_after_free()` — accesses memory after it is freed
- Fixed versions of each for comparison

---

## gsl::not_null — Document Intent

The [C++ Guidelines Support Library](https://github.com/microsoft/GSL) provides `gsl::not_null<T>`:

```cpp
#include <gsl/gsl>

void process(gsl::not_null<Model*> model) {
    // model is guaranteed non-null at this point
    model->run();  // safe — no null check needed
}

// Compile-time error or runtime assert if you try to pass null:
process(nullptr);  // Error!
```

This turns a comment ("this pointer must not be null") into a type-level
guarantee. The compiler enforces it.

For this course, `std::optional` and references cover most cases where you
would reach for `not_null`. But in large codebases, `gsl::not_null` is
invaluable for documenting interfaces.

---

## std::array vs std::vector — Stack vs Heap

### When Size Is Known at Compile Time

```cpp
// BBox always has exactly 4 values: x, y, w, h
struct BBox {
    std::array<double, 4> data;  // stack-allocated, zero overhead

    double x() const { return data[0]; }
    double y() const { return data[1]; }
    double w() const { return data[2]; }
    double h() const { return data[3]; }
};

// Kalman state is always 6 values: x, y, w, h, vx, vy
using KalmanState = std::array<double, 6>;
```

`std::array` vs `std::vector`:

| Feature | `std::array<T, N>` | `std::vector<T>` |
|---------|--------------------|--------------------|
| Size | Fixed at compile time | Dynamic |
| Allocation | Stack (zero cost) | Heap (malloc/free) |
| Cache | Always local | May be far away |
| Use when | Size is known | Size varies |

For fixed-size data like bounding boxes, Kalman states, and transformation
matrices, `std::array` eliminates heap allocation entirely. In a tracker
processing thousands of detections per frame, this is significant.

---

## Exercises

1. **Bounds-Checked Pipeline**: Modify `safe_views.cpp` to add a
   `span_dot_product()` function that computes the dot product of two spans.
   Return `std::optional<double>` — return `std::nullopt` if the spans have
   different sizes. Test it from Python.

2. **RAII File Reader**: Write an `RAIIFileReader` class that opens a file in
   its constructor and closes it in its destructor. Add a `read_doubles()`
   method that reads `n` doubles from the file into a `std::vector<double>`.
   Expose it via [nanobind](https://github.com/wjakob/nanobind). What happens if you try to read from a moved-from
   reader?

3. **Ownership Chain**: Create a chain of three objects where `A` uniquely owns
   `B` and `B` uniquely owns `C`. Transfer ownership of `A` from one variable
   to another using `std::move`. Verify that `B` and `C` move along with `A`.
   What is the state of the original variable after the move?

4. **ASAN Scavenger Hunt**: Add three more intentionally buggy functions to
   `asan_example.cpp`: a stack buffer overflow, a use-after-scope, and a
   double-free. Run with ASAN and read the reports. Write the fixed versions.

5. **Benchmark std::array vs std::vector**: Create 10,000 bounding boxes using
   `std::array<double, 4>` and 10,000 using `std::vector<double>(4)`. Measure
   the time to create them and the time to compute all areas. How much faster
   is the array version? Why?

## What You Learned

- `std::span` gives you safe, bounds-aware views with zero overhead in release
- `std::optional` eliminates null pointer bugs by encoding "might not exist" in the type
- RAII makes resource leaks impossible by tying cleanup to scope
- Smart pointers (`unique_ptr`, `shared_ptr`) make ownership explicit and automatic
- ASAN/UBSAN catch memory bugs that slip past the type system
- `std::array` avoids heap allocation for fixed-size data
- Modern C++ safety features are zero-cost — the safe code is often the fastest code

## Lesson Files

| File | Description |
|------|-------------|
| [safe_views.cpp](safe_views.cpp) | std::span safe view operations |
| [memory_safety_demo.cpp](memory_safety_demo.cpp) | RAII, optional, and smart pointers |
| [asan_example.cpp](asan_example.cpp) | Intentionally buggy code for ASAN demo |
| [safety_demo.py](safety_demo.py) | Python demo of safe views and memory safety |
| [CMakeLists.txt](CMakeLists.txt) | CMake build with ASAN/UBSAN options |
| [test_safety.py](test_safety.py) | Unit tests for all safety features |
| [test_integration_safety.py](test_integration_safety.py) | Integration test with safe tracking loop |
