# Section 11: Memory Safety Without Sacrifice

## Video 11.1: Why Memory Safety Matters (~8 min)

### Slides
- Slide 1: The Python safety net -- In tracker_engine, you never think about memory. Python handles allocation, deallocation, null checks, bounds checking. No segfaults, no double-frees, no buffer overflows. The cost is performance.
- Slide 2: Raw C++ is dangerous -- A single out-of-bounds write can corrupt memory silently, crash hours later, or appear to work in testing and explode in production. This lesson shows how modern C++ eliminates these bugs at compile time.
- Slide 3: The key insight -- Safety and performance are not opposites. The safest C++ patterns are often the fastest because they give the compiler more information to optimize. Zero-overhead safety is achievable.
- Slide 4: What we cover -- std::span (safe views), std::optional (no null pointers), RAII (automatic cleanup), smart pointers (clear ownership), ASAN/UBSAN (runtime bug detection), std::array (stack allocation).

### Key Takeaway
- Modern C++ safety features are zero-cost -- the safe code is often the fastest code because the compiler can reason about it more effectively.

## Video 11.2: std::span -- Safe Views Without Copies (~10 min)

### Slides
- Slide 1: The problem -- Raw pointer + size. `void process_row(const double* data, int size)` does not encode the relationship between data and size. Off-by-one errors produce undefined behavior with no error.
- Slide 2: std::span solution -- `void process_row(std::span<const double> data)` carries pointer and size together. Range-for loop cannot go out of bounds. Same two values as manual passing, but the compiler can reason about them.
- Slide 3: Debug vs release behavior -- Debug mode: throws `std::out_of_range` on bounds violation. Release mode (-O3): identical to raw pointer access, zero overhead, same assembly output.
- Slide 4: Slicing is free -- `std::span<const double> row = buffer.subspan(y * width, width)`. No copy, no allocation, just pointer arithmetic. The C++ equivalent of `buffer[y*width : y*width + width]` in Python.
- Slide 5: What we build -- `safe_views.cpp` provides `span_sum()`, `span_slice()`, `safe_at()`, and `raw_pointer_sum()` for comparison. In release mode, span_sum and raw_pointer_sum produce identical assembly.

### Key Takeaway
- std::span gives you bounds-aware views with zero runtime overhead in release builds -- the safety is free.

## Video 11.3: std::optional and RAII (~12 min)

### Slides
- Slide 1: The null pointer problem -- `BBox* find_target(const Frame&)` returns nullptr if no detection. Nothing forces the caller to check. Segfault when dereferencing null.
- Slide 2: std::optional solution -- `std::optional<BBox> find_target(const Frame&)` returns `std::nullopt` for no value. The type system forces you to check with `has_value()` or use `value_or(default)`. No heap allocation -- value stored inline with a boolean flag.
- Slide 3: RAII -- Resource Acquisition Is Initialization. Acquisition in the constructor, release in the destructor. C++ guarantees destructors run when objects leave scope, even during exceptions. Resources cannot leak.
- Slide 4: The Python context manager analogy -- `with open("file") as f:` ensures cleanup. RAII is the same pattern but automatic -- no `with` statement needed, no way to forget cleanup.
- Slide 5: RAII buffer example -- Constructor allocates, destructor frees. Cannot leak even if exceptions occur. The `RAIIBuffer` class demonstrates this pattern with nanobind bindings.

### Key Takeaway
- std::optional encodes "might not exist" in the type system (no null pointer crashes), and RAII ties resource cleanup to scope (no resource leaks).

## Video 11.4: Smart Pointers and Ownership (~10 min)

### Slides
- Slide 1: The ownership problem -- `Model* load_model(path)` returns a raw pointer. Who deletes it? If nobody: memory leak. If both tracker and pipeline delete: double-free crash.
- Slide 2: unique_ptr -- Exactly one owner. `auto model = std::make_unique<Model>("yolov8.onnx")`. Transfer with `std::move(model)` -- original is now nullptr. Cannot accidentally use after move.
- Slide 3: shared_ptr -- Multiple owners, reference counted. `auto buffer = std::make_shared<Buffer>(1024)`. Deleted when last shared_ptr goes out of scope. Similar to Python's reference counting but explicit in the type.
- Slide 4: Ownership semantics in types -- `unique_ptr<T>` = "I own this exclusively", `shared_ptr<T>` = "multiple owners, last one cleans up", `T&` = "borrowing, someone else owns it", `T*` = "non-owning observer, do not delete".
- Slide 5: Jetson note -- On memory-constrained Jetson devices, smart pointers help prevent memory leaks in long-running inference pipelines. A tracker that leaks even small amounts per frame will eventually exhaust Jetson's limited RAM.

### Key Takeaway
- Smart pointers encode ownership in the type system -- unique_ptr for exclusive ownership, shared_ptr for shared ownership, and the compiler enforces the rules.

## Video 11.5: ASAN, UBSAN, and Compile-Time Safety (~10 min)

### Slides
- Slide 1: AddressSanitizer (ASAN) catches -- Buffer overflow, use-after-free, memory leaks, double-free, stack buffer overflow. UndefinedBehaviorSanitizer (UBSAN) catches -- Signed integer overflow, null pointer dereference, misaligned access.
- Slide 2: Enabling in CMake -- `add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)`. Build with `-DENABLE_SANITIZERS=ON -DCMAKE_BUILD_TYPE=Debug`.
- Slide 3: Reading ASAN output -- The report tells you what happened (heap-buffer-overflow), where (file:line), where memory was allocated, the exact byte offset. Detailed stack traces for both the error and the allocation.
- Slide 4: std::array vs std::vector -- When size is known at compile time, use `std::array<double, 4>` (stack-allocated, zero cost, always local in cache) instead of `std::vector<double>` (heap-allocated, malloc/free overhead). BBox with 4 values, Kalman state with 6 values -- always fixed size.
- Slide 5: gsl::not_null -- Documents that a pointer must not be null. `process(gsl::not_null<Model*> model)` turns a comment into a type-level guarantee. Useful in large codebases for documenting interfaces.

### Live Demo
- Build `asan_example.cpp` with sanitizers enabled. Run the buggy versions and walk through the ASAN reports. Then run the fixed versions showing clean output.

### Key Takeaway
- ASAN and UBSAN are your safety net during development -- enable them in debug builds to catch memory bugs that slip past the type system.
