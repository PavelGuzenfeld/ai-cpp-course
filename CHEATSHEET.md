# C++ for CV/AI — Quick Reference

## Memory Hierarchy

```
L1 cache:   0.5 ns    32-64 KB     ← Keep hot data here
L2 cache:     7 ns    256 KB-1 MB
L3 cache:    20 ns    8-32 MB
RAM:        100 ns    16-64 GB     ← 200x slower than L1
SSD:    100,000 ns    TB
PCIe:    ~80 ns/B     12 GB/s      ← GPU transfer bottleneck
GPU mem:   ~1 ns/B    900 GB/s     ← 75x faster than PCIe
```

## Python Optimization Patterns

```python
# __slots__ — eliminate __dict__ (2x less memory)
class BBox:
    __slots__ = ('x', 'y', 'w', 'h')

# dataclass with slots (Python 3.10+)
@dataclass(slots=True)
class BBox:
    x: float; y: float; w: float; h: float

# numpy view (zero-copy)
view = buf[start:end]          # no allocation
copy = buf[start:end].copy()   # allocates

# Pre-allocated buffers
np.subtract(a, b, out=result)  # reuse result array

# Thread pool (not Thread-per-task)
pool = ThreadPoolExecutor(max_workers=4)
pool.submit(save_image, path, data)

# torch.inference_mode (faster than no_grad)
with torch.inference_mode():   # not torch.no_grad()
    output = model(input)
```

## Nanobind Binding Patterns

```cpp
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
namespace nb = nanobind;

// Bind a class
NB_MODULE(my_module, m) {
    nb::class_<MyClass>(m, "MyClass")
        .def(nb::init<int>(), nb::arg("value"))
        .def_rw("x", &MyClass::x)              // read-write
        .def_prop_ro("area", &MyClass::area)    // computed property
        .def("method", &MyClass::method);
}

// Zero-copy numpy return
nb::ndarray<nb::numpy, float> result(ptr, ndim, shape, owner);

// CMake
find_package(nanobind CONFIG REQUIRED)
nanobind_add_module(my_module NB_STATIC source.cpp)
```

## C++ Safety Patterns

```cpp
// std::span — safe view (zero-cost in release)
void process(std::span<const double> data) {
    for (auto val : data) { /* bounds-safe iteration */ }
}

// std::optional — no null pointers
std::optional<BBox> detect(const Image& img) {
    if (found) return BBox{x, y, w, h};
    return std::nullopt;
}
if (auto det = detect(img)) {
    use(det->x);  // safe — checked
}

// RAII — resources clean themselves
{
    auto buf = std::make_unique<float[]>(size);
    // use buf...
}  // automatically freed here, even if exception

// Concepts — constrain templates
template <typename T>
concept FlatType = std::is_trivially_copyable_v<T>
               && std::is_standard_layout_v<T>;

template <FlatType T>
void serialize(const T& obj) { memcpy(buf, &obj, sizeof(T)); }
```

## GPU Rules of Thumb

```
1. Minimize CPU↔GPU transfers (PCIe is the bottleneck)
2. Fuse operations (one kernel > three kernels)
3. Pre-allocate everything (no per-frame malloc)
4. Use pinned memory (2x faster transfers)
5. Batch inference (fixed overhead × 1 < fixed overhead × N)
6. Keep data on GPU as long as possible
7. Use CUDA streams to overlap transfer + compute
```

```python
# Pinned memory
tensor = torch.empty(shape, pin_memory=True)
gpu_tensor.copy_(tensor, non_blocking=True)

# CUDA timing (not CPU timers!)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
output = model(input)
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end)
```

## Compile-Time Patterns

```cpp
// constexpr LUT — computed at compile time
constexpr auto LUT = make_grayscale_lut();  // in .rodata

// variant state machine — type-safe, zero-overhead
using State = std::variant<Idle, Tracking, Lost, Search>;
state = std::visit(overloaded{
    [](Idle) -> State { ... },
    [](Tracking& t) -> State { ... },
    // compiler error if you forget a state
}, state);

// if constexpr — dead branch elimination
template <Mode M>
void process() {
    if constexpr (M == Mode::Fast) { /* only this compiled */ }
    else { /* eliminated */ }
}
```

## Profiling Workflow

```
1. MEASURE  → time.perf_counter_ns() / torch.cuda.Event
2. IDENTIFY → which stage dominates? (Amdahl's law)
3. CHOOSE   → technique from this cheatsheet
4. IMPLEMENT → minimal change
5. MEASURE  → did it actually help?
6. REPEAT   → until Amdahl says stop
```

## Packaging

```toml
# pyproject.toml (scikit-build-core + nanobind)
[build-system]
requires = ["scikit-build-core", "nanobind"]
build-backend = "scikit_build_core.build"

[project]
name = "my-package"
version = "0.1.0"
requires-python = ">=3.8"
```

```bash
pip install .          # install from source
pip wheel . -w dist/   # build wheel
```

## Build Commands

```bash
# Docker
docker build -t ai-cpp-course -f Dockerfile .
docker run -it -v $(pwd):/workspace ai-cpp-course

# Build all
colcon build && source install/setup.bash

# Test
pytest ai-cpp-l5/ -v        # Python-only
pytest ai-cpp-l4/ -v        # after colcon build

# Sanitizers
cmake -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined" ..
```
