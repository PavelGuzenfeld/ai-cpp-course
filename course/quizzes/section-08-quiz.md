# Section 8 Quiz: Compile-Time Concepts for Performance

## Q1: What problem do C++20 concepts solve compared to unconstrained templates?

- a) They make templates run faster at runtime
- b) They provide readable, compile-time constraints on template parameters with clear error messages, replacing the cryptic SFINAE technique
- c) They allow templates to work with Python types
- d) They enable templates to be used across different compilers

**Answer: b)** Before concepts, constraining templates required SFINAE, which produced incomprehensible error messages. Concepts let you write constraints like `template <FlatType T>` that produce clear errors such as "constraint not satisfied: FlatType" when violated.

## Q2: Why does a `constexpr` lookup table (LUT) have zero runtime cost?

- a) The LUT is stored on the GPU
- b) The compiler evaluates the function at compile time and embeds the results directly in the binary's `.rodata` section -- no runtime computation or initialization is needed
- c) The LUT is computed lazily on first access
- d) The operating system precomputes it during boot

**Answer: b)** A `constexpr` function is evaluated during compilation. The resulting array is placed in the read-only data section of the binary, ready to use immediately at program start with no runtime cost.

## Q3: A tracker uses string comparisons for state transitions (`if self.state == "tracking"`). What problem does `std::variant` solve here?

- a) It makes the string comparisons faster
- b) It provides type-safe states where each state can carry its own data, the compiler enforces exhaustive handling, and dispatch is a zero-cost jump table instead of string hashing
- c) It encrypts the state names for security
- d) It allows states to be serialized to JSON

**Answer: b)** With `std::variant<Idle, Tracking, Lost, Search>`, each state is a distinct type that can hold state-specific data (e.g., `Lost` carries `frames_lost`). The compiler forces you to handle every state in `std::visit`, and a typo like `Trakcing` is a compile error, not a silent runtime bug.

## Q4: What does `if constexpr` accomplish that a regular `if` statement cannot?

- a) It evaluates conditions in parallel
- b) It eliminates the untaken branch entirely at compile time, so only the selected code path exists in the final binary
- c) It makes the condition check faster at runtime
- d) It allows conditions based on string values

**Answer: b)** `if constexpr` evaluates the condition at compile time. The compiler discards the branch that is not taken, meaning it produces no machine code and incurs no runtime cost -- unlike a regular `if`, which must evaluate the condition each time.

## Q5: Why is template-based dispatch preferred over virtual functions when the type is known at compile time?

- a) Templates use less memory
- b) Template dispatch resolves at compile time, allowing the compiler to inline the function call and avoid vtable pointer indirection and potential cache misses
- c) Virtual functions do not work with modern compilers
- d) Templates are required by the C++ standard for performance-critical code

**Answer: b)** Virtual function dispatch requires loading a vtable pointer, dereferencing it to find the function address, and calling through that pointer -- which prevents inlining and may cause a cache miss. Template dispatch resolves the concrete function at compile time, enabling full inlining.

## Q6: What would happen if you tried to instantiate a `FlatType`-constrained template with `std::vector<int>`?

- a) It would work correctly because vectors are contiguous
- b) The program would crash at runtime with a segfault
- c) The compiler would reject it with a clear constraint-violation error because `std::vector` is not trivially copyable
- d) It would compile but produce incorrect results

**Answer: c)** `std::vector` manages heap-allocated memory through internal pointers, so it is not trivially copyable. The `FlatType` concept check fails at compile time with a clear error, preventing you from putting a vector into shared memory or copying it with `memcpy`.

## Q7: How does strong typing with template tags prevent bugs in configuration APIs?

- a) It adds runtime type checks to all function calls
- b) It creates distinct types (e.g., `FrameCount` vs `Duration`) so the compiler prevents accidentally passing one where the other is expected
- c) It encrypts configuration values
- d) It forces all values to be strings

**Answer: b)** Template tags like `SanitizedKey<FrameCountTag>` and `SanitizedKey<DurationTag>` are different types even though they wrap the same underlying data. Passing a `FrameCount` where a `Duration` is expected is a compile error, catching unit-mismatch bugs that stringly-typed APIs cannot.

## Q8: In the tracker_engine pattern `crop_and_resize(img, bbox, size, mode='bilinear')`, what is the performance cost of the string-based mode parameter?

- a) No cost -- string comparison is free
- b) Each call pays for string hashing and equality comparison, even though the mode is typically fixed for the lifetime of the application
- c) The string must be sent over the network for validation
- d) String parameters cause memory leaks

**Answer: b)** Every invocation compares the mode string character by character against known values. When the mode is fixed at configuration time, this work is repeated unnecessarily on every frame. A compile-time enum or template parameter resolves the mode once with zero per-call overhead.
