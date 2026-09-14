# Lesson 13: Ownership of a C Handle — RAII, Move-Only, and the Borrowed-Pointer Double-Free

## Goal

Python's refcounting has no analogue for "who owns this" — every object has
exactly one implicit owner (the interpreter) and the question never comes up.
A C API built around opaque handles (`handle_t create(); void destroy(handle_t);`)
has no such guarantee: any function can be handed a handle without knowing
whether it owns it. This lesson teaches the RAII wrapper that answers that
question at compile time, and the bug that appears when the wrapper answers
it wrong.

## The C API Stand-In

```cpp
namespace c_api {
    using handle_t = std::uint64_t;
    handle_t create();
    void destroy(handle_t) noexcept;
}
```

`destroy()` is instrumented to count a **double-destroy** — a call on a
handle that was already destroyed — so tests observe the bug directly
instead of relying on a sanitizer or a crash.

## `OwnedHandle`: Move-Only RAII

```cpp
class OwnedHandle {
public:
    OwnedHandle();                         // creates and owns a new handle
    OwnedHandle(handle_t h, bool owns);    // wrap an existing handle
    OwnedHandle(OwnedHandle const&) = delete;
    OwnedHandle(OwnedHandle&& other) noexcept;   // steal-and-null
    ~OwnedHandle();                        // destroys only if it owns

    handle_t get() const noexcept;         // borrow: look without taking
    handle_t release() noexcept;           // hand ownership back out
};
```

- **Copy is deleted.** Two owners of the same handle is the bug this type
  exists to prevent, so the compiler refuses it outright.
- **Move steals and nulls the source.** After `b = std::move(a)`, `a` no
  longer owns anything — its destructor becomes a no-op.
- **`release()` is the escape hatch.** A function that must hand a handle to
  code outside RAII's reach (a callback, a C API, a caller who will manage it
  itself) calls `release()` to opt out of the destructor without leaking.

## The Bug: an Owning Wrapper Around a Borrowed Pointer

From `PRODUCTION_PLAN.md` §1.1: a transform function wrapped a **non-owned**
pointer in the RAII type. When the wrapper went out of scope, its destructor
freed memory the pipeline still owned. The fix was `release()` — but only
once someone noticed the destructor was firing on memory it never allocated.

```cpp
// BUG: `h` is borrowed from a pool the caller does not own.
OwnedHandle wrap_borrowed_incorrectly(handle_t h) {
    return OwnedHandle(h, /*takes_ownership=*/true);   // wrong
}

// FIX: wrap it as non-owning. Its destructor is a no-op.
OwnedHandle wrap_borrowed_correctly(handle_t h) {
    return OwnedHandle(h, /*takes_ownership=*/false);  // right
}
```

The symptom is delayed: the buggy wrapper's destructor runs and frees the
handle immediately, but nothing crashes. The pool that still thinks it owns
the handle only discovers the problem later, when *it* tries to free the
same id — often long after the buggy call site has scrolled off the screen.
`test_integration_ownership.py::test_one_buggy_transform_corrupts_the_pool`
reproduces exactly this: the double-destroy count only goes non-zero when
the pool is cleared, not when the buggy transform runs.

## Borrow vs. Own at a Function Boundary

- **Borrow**: take a raw `handle_t` (or `OwnedHandle const&`). The function
  may read through it but must not outlive the call, and must never wrap it
  as owning.
- **Own**: take an `OwnedHandle` by value (forcing the caller to `std::move`
  it in) or return one. The function is now responsible for its lifetime.

A function signature that takes a raw `handle_t` is documentation: it says
"I am borrowing this." A function that takes `OwnedHandle&&` says "I am
taking this from you." Losing that distinction is exactly how the production
bug happened.

## Build and Run

```bash
cd /workspace
colcon build --packages-select nanobind-l13
source install/setup.bash

pytest ai-cpp-l13/ -v
```

## What You Learned

- An opaque C handle has no owner unless something says so explicitly
- `OwnedHandle` makes ownership move-only: copy is deleted, move steals and
  nulls the source
- `release()` is the correct way to hand ownership out without destroying
- Wrapping a *borrowed* pointer as *owned* is a double-free, and the symptom
  can appear far from the bug — when the real owner tries to free it later
- A function's parameter type (`handle_t` vs `OwnedHandle&&`) documents
  borrow vs. own; losing that distinction is how the production bug happened

## Exercises

1. **Trigger the bug without the helper**: call `wrap_borrowed_incorrectly`
   yourself on a handle from a hand-rolled pool of three `OwnedHandle`s, and
   find the double-destroy without looking at `test_integration_ownership.py`
   first.

2. **Add a `swap()` method**: implement `OwnedHandle::swap(OwnedHandle&)` and
   write a test that exchanges ownership between two wrappers without ever
   calling `destroy()`.

3. **Const-correctness**: why does `get()` work on a `const OwnedHandle&` but
   `release()` cannot? Try marking `release()` const and read the compiler
   error.

4. **Python holds the last reference**: construct an `OwnedHandle` in Python,
   store it in a list, `del` the list, and confirm (via `live_count()`)
   that nanobind's holder called the C++ destructor.

## Lesson Files

| File | Description |
|------|-------------|
| [ownership_native.cpp](ownership_native.cpp) | Mock C API, `OwnedHandle` RAII wrapper, nanobind bindings |
| [CMakeLists.txt](CMakeLists.txt) | CMake build configuration |
| [test_ownership.py](test_ownership.py) | Unit tests: lifecycle, move, release, the double-free and its fix |
| [test_integration_ownership.py](test_integration_ownership.py) | Integration tests: a pool of owned handles borrowed by transform stages |
