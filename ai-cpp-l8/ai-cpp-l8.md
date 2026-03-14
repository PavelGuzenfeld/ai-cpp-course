# Lesson 8: Compile-Time Concepts for Performance

## Why Compile-Time Matters for CV/Tracking

Python developers often treat types as runtime metadata — strings, dicts, isinstance checks.
C++ flips this: the compiler knows types, sizes, and constraints at compile time. This lesson
shows how to exploit that knowledge for zero-overhead abstractions that catch bugs before
your code ever runs.

The [tracker_engine](https://github.com/thebandofficial/tracker_engine) codebase is riddled
with patterns that pay runtime costs for things the compiler could resolve statically:
- State machines using string comparison (`state in ['lost', 'search']`)
- Mode selection via string arguments (`crop_and_resize(..., mode='bilinear')`)
- Type checks at runtime instead of compile time

## C++20 Concepts: Constrained Templates

Before C++20, constraining templates required SFINAE — a technique so ugly it became a meme.
Concepts replace SFINAE with readable, composable constraints.

### The Problem with Unconstrained Templates

```cpp
template <typename T>
void serialize(T* data, size_t size) {
    memcpy(buffer, data, size);  // What if T has pointers? Dangling refs in shared memory.
}
```

If someone passes a `std::string`, this compiles but produces garbage. The error message
when things go wrong is a wall of template instantiation noise.

### Concepts Fix This

```cpp
template <typename T>
concept FlatType = std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>;

template <FlatType T>
void serialize(T* data, size_t size) {
    memcpy(buffer, data, size);  // Compiler guarantees T is safe to memcpy
}
```

Now `serialize<std::string>(...)` produces a clear error: "constraint not satisfied: FlatType".

### Why FlatType Matters

A `FlatType` is a type that can be safely:
- **Copied with `memcpy`**: No copy constructors, no pointer fixups
- **Mapped into shared memory**: Layout matches across processes
- **Sent to a GPU**: No pointer chasing, contiguous bytes
- **Serialized to disk**: What you write is what you read

The safe-shm submodule in Lesson 3 already uses this concept to prevent you from putting
a `std::vector` into shared memory (which would crash when another process tries to read it).

### Composing Concepts

```cpp
template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

template <typename T>
concept ImageLike = requires(T img) {
    { img.data() } -> std::convertible_to<const void*>;
    { img.width() } -> std::convertible_to<int>;
    { img.height() } -> std::convertible_to<int>;
    { img.channels() } -> std::convertible_to<int>;
};
```

Concepts can be combined with `&&` and `||`, and used with `requires` clauses to express
arbitrary structural constraints.

## Compile-Time Image Dimensions

### The Python Way (Runtime Everything)

```python
def resize(img, target_w, target_h):
    assert img.shape == (1080, 1920, 3)  # Runtime check, every call
    # ...
```

### The C++ Way (Compile-Time Dimensions)

```cpp
template <int W, int H, int C>
struct Image {
    static constexpr int width = W;
    static constexpr int height = H;
    static constexpr int channels = C;
    static constexpr size_t size = W * H * C;

    std::array<uint8_t, size> data;
};

using HD_RGB = Image<1920, 1080, 3>;
using HD_Gray = Image<1920, 1080, 1>;
```

Benefits:
- **Size known at compile time**: `sizeof(HD_RGB)` is a constant, no runtime query
- **Static assertions**: `static_assert(HD_RGB::channels == 3)` — fails at compile time
- **Zero overhead**: No virtual dispatch, no heap allocation, no runtime checks
- **Stack allocation**: Small images live on the stack, no `new`/`delete`

## `constexpr` LUT Generation

Lookup tables (LUTs) are a classic optimization: precompute results instead of computing
them on every pixel. In Python, you'd compute the LUT at module import time. In C++,
you can compute it at *compile time* — the LUT exists in the binary, ready to use.

### BGR-to-Grayscale LUT

```cpp
constexpr auto make_grayscale_lut() {
    std::array<uint8_t, 256 * 3> lut{};
    for (int i = 0; i < 256; ++i) {
        lut[i]       = static_cast<uint8_t>(i * 0.114);  // B
        lut[256 + i] = static_cast<uint8_t>(i * 0.587);  // G
        lut[512 + i] = static_cast<uint8_t>(i * 0.299);  // R
    }
    return lut;
}

constexpr auto GRAY_LUT = make_grayscale_lut();
```

The compiler evaluates `make_grayscale_lut()` during compilation. At runtime, `GRAY_LUT`
is just a read from a pre-filled array in the `.rodata` section.

### Gamma Correction LUT

```cpp
constexpr auto make_gamma_lut(double gamma) {
    std::array<uint8_t, 256> lut{};
    for (int i = 0; i < 256; ++i) {
        double normalized = i / 255.0;
        double corrected = /* pow approximation */;
        lut[i] = static_cast<uint8_t>(corrected * 255.0);
    }
    return lut;
}
```

Note: `std::pow` is not `constexpr` in most implementations, so we use a polynomial
approximation or iterative method for compile-time evaluation.

## `std::variant` + `std::visit` vs String-Based State Machines

### The tracker_engine Pattern (Runtime Strings)

```python
class Tracker:
    def __init__(self):
        self.state = "idle"

    def update(self, detection):
        if self.state == "idle":
            if detection:
                self.state = "tracking"
        elif self.state == "tracking":
            if not detection:
                self.state = "lost"
        elif self.state == "lost":
            self.lost_frames += 1
            if self.lost_frames > 30:
                self.state = "search"
```

Problems:
- Typo `"trakcing"` compiles and runs — fails silently
- Each state comparison is a string hash + equality check
- No way to enforce that each state carries the right data

### The C++ Way (Type-Safe States)

```cpp
struct Idle {};
struct Tracking { BBox target; };
struct Lost { int frames_lost; };
struct Search { BBox last_known; };

using TrackerState = std::variant<Idle, Tracking, Lost, Search>;
```

Transitions use `std::visit` with an overloaded lambda:

```cpp
TrackerState next = std::visit(overloaded{
    [&](Idle) -> TrackerState {
        return detection ? TrackerState{Tracking{*detection}} : TrackerState{Idle{}};
    },
    [&](Tracking& t) -> TrackerState {
        return detection ? TrackerState{Tracking{*detection}} : TrackerState{Lost{1}};
    },
    [&](Lost& l) -> TrackerState {
        if (detection) return TrackerState{Tracking{*detection}};
        return l.frames_lost > 30 ? TrackerState{Search{l.last_known}}
                                  : TrackerState{Lost{l.frames_lost + 1}};
    },
    [&](Search& s) -> TrackerState {
        return detection ? TrackerState{Tracking{*detection}} : TrackerState{Search{s.last_known}};
    }
}, state);
```

Benefits:
- **Compile-time exhaustiveness**: Add a new state, forget a handler? Compiler error.
- **Each state carries its own data**: `Lost` has `frames_lost`, `Tracking` has `target`.
- **Zero-overhead dispatch**: `std::visit` compiles to a jump table — no string hashing.
- **No typo bugs**: `Trakcing` is a compile error, not a silent bug.

## `if constexpr`: Zero-Cost Branching

### Runtime String Comparison (tracker_engine)

```python
def crop_and_resize(img, bbox, size, mode='bilinear'):
    if mode == 'bilinear':
        return cv2.resize(crop, size, interpolation=cv2.INTER_LINEAR)
    elif mode == 'nearest':
        return cv2.resize(crop, size, interpolation=cv2.INTER_NEAREST)
```

Every call pays for string comparison even though the mode is usually fixed.

### Compile-Time Selection

```cpp
enum class Interp { Bilinear, Nearest };

template <Interp Mode>
void resize(const Image& src, Image& dst) {
    if constexpr (Mode == Interp::Bilinear) {
        // bilinear code — only this branch is compiled
    } else if constexpr (Mode == Interp::Nearest) {
        // nearest code — only this branch is compiled
    }
}
```

The compiler eliminates the unused branch entirely. The resulting binary contains only
the code for the mode you actually use.

## Strong Typing with Templates

### The Problem

```cpp
void set_config(const std::string& key, const std::string& value);
set_config("max_age", "30");  // Is this seconds? frames? meters?
```

### Template Tags

```cpp
template <typename Tag>
struct SanitizedKey {
    std::string value;
    explicit SanitizedKey(std::string v) : value(std::move(v)) {}
};

struct FrameCountTag {};
struct DurationTag {};

using FrameCount = SanitizedKey<FrameCountTag>;
using Duration = SanitizedKey<DurationTag>;

void set_max_age(FrameCount count);  // Can't accidentally pass a Duration
```

## Template-Based Dispatch vs Virtual Functions

Virtual functions have overhead:
- Pointer indirection through vtable
- Prevents inlining
- Cache miss on vtable lookup

Template-based dispatch resolves at compile time:

```cpp
// Virtual (runtime dispatch)
struct Detector {
    virtual std::vector<BBox> detect(const Image& img) = 0;
};

// Template (compile-time dispatch)
template <typename DetectorT>
void track(DetectorT& detector, const Image& img) {
    auto boxes = detector.detect(img);  // Inlined, no vtable
}
```

Use virtual functions when you need runtime polymorphism (plugin systems, user-selected
algorithms). Use templates when the type is known at compile time.

## Build and Run

```bash
# Inside Docker container
cd /workspace
colcon build --packages-select nanobind-l8
source install/setup.bash

# Run benchmarks
python3 ai-cpp-l8/benchmark_concepts.py

# Run the Python state machine demo
python3 ai-cpp-l8/state_machine_slow.py

# Run tests
pytest ai-cpp-l8/ -v
```

## What You Learned

- C++20 concepts replace SFINAE with readable type constraints
- `FlatType` (trivially_copyable + standard_layout) gates safe memcpy/shm/GPU operations
- `constexpr` LUTs compute at compile time — zero runtime cost, embedded in `.rodata`
- `std::variant` + `std::visit` is a type-safe, zero-overhead alternative to string state machines
- `if constexpr` eliminates dead branches at compile time — only the selected path exists
- Strong typing with template tags prevents accidentally mixing unrelated values
- Template dispatch resolves at compile time; virtual dispatch resolves at runtime

## Exercises

1. **Concept definition**: Add a `Serializable` concept that requires both `FlatType` and
   a `serialize()` method. Test it with valid and invalid types.

2. **LUT extension**: Create a compile-time LUT for HSV-to-RGB conversion. Benchmark it
   against OpenCV's `cvtColor`.

3. **State machine extension**: Add a `Paused` state to the variant state machine. Verify
   the compiler forces you to handle it in every `visit` call.

4. **`if constexpr` pipeline**: Create a template image processing pipeline where each
   stage (grayscale, blur, threshold) is selected at compile time.

## Files in This Lesson

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Build configuration for all nanobind modules |
| `concepts_demo.cpp` | FlatType, Numeric, ImageLike concepts with constrained templates |
| `compile_time_lut.cpp` | constexpr LUT generation for grayscale and gamma correction |
| `state_machine.cpp` | String-based vs variant-based state machine comparison |
| `state_machine_slow.py` | Python string-based state machine (the "before" version) |
| `benchmark_concepts.py` | Performance comparison across all techniques |
| `test_concepts.py` | Unit tests for concepts, LUTs, and state machine |
| `test_integration_concepts.py` | Integration tests simulating real tracking workloads |
