# Section 8: Compile-Time Concepts for Performance

## Video 8.1: Why Compile-Time Matters (~8 min)

### Slides
- Slide 1: Python treats types as runtime metadata -- String comparisons for state machines (`state in ['lost', 'search']`), mode selection via string arguments, isinstance checks at runtime. C++ flips this: the compiler knows types, sizes, and constraints at compile time.
- Slide 2: tracker_engine examples -- State machine using string comparison (typo "trakcing" silently fails), crop_and_resize with string mode argument, type checks scattered throughout.
- Slide 3: The compile-time advantage -- Bugs caught before code runs, zero-overhead dispatch (no string hashing), dead code elimination, the compiler removes unused branches entirely.
- Slide 4: What we cover -- C++20 concepts (constrained templates), constexpr LUT generation, std::variant state machines, if constexpr branching, template-based dispatch.

### Key Takeaway
- Moving checks from runtime to compile time catches bugs earlier and eliminates overhead -- the compiler does the work so your CPU does not have to.

## Video 8.2: C++20 Concepts (~12 min)

### Slides
- Slide 1: The problem with unconstrained templates -- `template <typename T> void serialize(T* data, size_t size)` with `memcpy` compiles for `std::string` but produces garbage. Error messages are walls of template instantiation noise.
- Slide 2: Concepts fix this -- `concept FlatType = std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>`. Now `template <FlatType T> void serialize(...)` produces a clear error: "constraint not satisfied."
- Slide 3: Why FlatType matters -- Safe to memcpy, safe for shared memory, safe for GPU transfer, safe for disk serialization. Used in Lesson 3's safe-shm to prevent putting std::vector into shared memory.
- Slide 4: Composing concepts -- `concept Numeric = std::is_arithmetic_v<T>`. `concept ImageLike = requires(T img) { img.data(), img.width(), img.height(), img.channels() }`. Combine with && and ||, use requires clauses.
- Slide 5: Compile-time image dimensions -- `template <int W, int H, int C> struct Image` with `static constexpr` size. `using HD_RGB = Image<1920, 1080, 3>`. Size known at compile time, zero overhead, stack allocation for small images.

### Key Takeaway
- C++20 concepts replace SFINAE with readable, composable type constraints that produce clear error messages when violated.

## Video 8.3: constexpr LUTs and if constexpr (~10 min)

### Slides
- Slide 1: Compile-time lookup tables -- In Python, compute LUTs at import time. In C++, compute at compile time with `constexpr`. The LUT exists in the binary's `.rodata` section, ready to use with zero runtime cost.
- Slide 2: BGR-to-grayscale LUT example -- `constexpr auto make_grayscale_lut()` computes 256*3 entries at compile time. At runtime, grayscale conversion is a single table lookup per pixel.
- Slide 3: Gamma correction LUT -- `constexpr auto make_gamma_lut(double gamma)` uses polynomial approximation since `std::pow` is not constexpr. The compiler evaluates the entire function during compilation.
- Slide 4: if constexpr -- Zero-cost branching. `template <Interp Mode> void resize(...)` with `if constexpr (Mode == Interp::Bilinear)`. The compiler eliminates the unused branch entirely -- the binary contains only the code for the selected mode.
- Slide 5: Comparison to Python -- `if mode == 'bilinear':` pays for string comparison every call even when mode is fixed. `if constexpr` makes the decision at compile time -- the runtime code has no branch at all.

### Key Takeaway
- constexpr LUTs embed precomputed data in the binary with zero runtime cost, and if constexpr eliminates unused code paths entirely at compile time.

## Video 8.4: Variant State Machines and Template Dispatch (~12 min)

### Slides
- Slide 1: tracker_engine's string state machine -- `self.state = "idle"`, `if self.state == "tracking"`. Typos compile and run silently. Each comparison is a string hash + equality check. No enforcement that each state carries the right data.
- Slide 2: std::variant state machine -- `struct Idle {}; struct Tracking { BBox target; }; struct Lost { int frames_lost; };`. `using TrackerState = std::variant<Idle, Tracking, Lost, Search>;`. Each state carries its own data.
- Slide 3: std::visit with overloaded lambda -- Pattern matching on states. Compile-time exhaustiveness checking: add a new state, forget a handler, get a compiler error. Zero-overhead dispatch via jump table.
- Slide 4: Template dispatch vs virtual functions -- Virtual functions have pointer indirection through vtable, prevent inlining, cache miss on vtable lookup. Template dispatch resolves at compile time, allows inlining, no vtable. Use virtual when type is unknown at compile time (plugins), templates when type is known.
- Slide 5: Strong typing with template tags -- `SanitizedKey<FrameCountTag>` vs `SanitizedKey<DurationTag>`. Cannot accidentally mix frame counts and durations. The type system prevents the bug at compile time.

### Live Demo
- Run `benchmark_concepts.py` comparing string state machine vs variant state machine performance. Show the compiler error when a state handler is missing.

### Key Takeaway
- std::variant + std::visit is a type-safe, zero-overhead alternative to string-based state machines -- the compiler enforces exhaustive handling of all states.
