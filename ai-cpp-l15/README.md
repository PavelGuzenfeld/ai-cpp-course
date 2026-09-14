# Lesson 15: Threading and Atomics — What the GIL Was Hiding

## Goal

The GIL makes Python's threading module look almost pointless: one thread
runs Python bytecode at a time, so a "race" between two Python threads is
rarely a memory-corruption problem. C++ has no such guarantee. This lesson
is the concurrency content the course never had: `std::atomic`,
`memory_order`, a real data race, and what `nb::gil_scoped_release` actually
buys you.

## Data Race vs. Race Condition

A **data race** is two threads accessing the same memory, at least one a
write, with no ordering between them — undefined behaviour, full stop. A
**race condition** is a program whose *outcome* depends on timing, which can
happen even with no data race (e.g. two well-synchronized threads racing to
acquire a lock). This lesson is about the first kind.

## `SpscRing`: a Single-Producer/Single-Consumer Ring

```cpp
bool push(T const& value) noexcept {
    std::size_t const head = head_.load(std::memory_order_relaxed);
    std::size_t const next = (head + 1) & (N - 1);
    if (next == tail_.load(std::memory_order_acquire)) return false;
    buffer_[head] = value;                       // write the slot first
    head_.store(next, std::memory_order_release); // then publish it
    return true;
}
```

The publish sequence is: **write the data, then publish the index.** A
consumer that observes the new `head` with an `acquire` load is guaranteed
by the C++ memory model to see everything the producer wrote before the
matching `release` store — including `buffer_[head]`.

## The Bug: Publishing Before Writing

`BrokenSpscRing` swaps the order:

```cpp
head_.store(next, std::memory_order_release); // BUG: published first
buffer_[head] = value;                         // consumer may read this now
```

This is the bug class from `PRODUCTION_PLAN.md` §3.4: a `ready` flag (or
here, the ring index) published before the data it announces. A consumer
that sees the new head may read `buffer_[head]` while the producer is still
writing it, and gets whatever was in that slot from a previous lap of the
ring instead. `BrokenSpscRing::push` adds a deliberate 20µs delay between
the two statements — not realistic, but it widens the race window so the
bug reproduces on every run instead of depending on timing luck (the same
reasoning L19 uses an explicit barrier for).

`test_concurrency.py` and `test_integration_concurrency.py` both observe
this directly: the correct ring always delivers `list(range(n))`, the
broken one reliably does not.

## Why `volatile` Is Not `std::atomic`

`volatile` tells the compiler not to elide or reorder accesses to a
variable *relative to other volatile accesses* — it says nothing about
inter-thread visibility or ordering, and provides no atomicity for
multi-byte types. `std::atomic<T>` is the only standard tool that gives you
both: guaranteed atomicity and a specified memory-ordering relationship
(`memory_order_relaxed`/`acquire`/`release`/`seq_cst`) with other threads.

## `std::mutex` and When a Lock Is the Right Answer

`SpscRing` avoids a mutex because a single producer and single consumer
never contend for the same operation (push vs. pop touch different
indices). The moment you have more than one producer or more than one
consumer, that stops being true, and a lock (or a proper MPMC structure) is
the honest answer — a lock-free structure retrofitted for multiple writers
without redesigning it is how these bugs get shipped.

## `nb::gil_scoped_release`

```cpp
void busy_wait_ms(int ms, bool release_gil) {
    if (release_gil) {
        nb::gil_scoped_release release;
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }
}
```

Without releasing the GIL, a C++ call blocks every other Python thread for
its entire duration — the "concurrency" is fake. `test_concurrency.py`
proves it with a wall-clock measurement: a background Python thread makes
almost no progress while the GIL is held, and orders of magnitude more once
it is released.

## ThreadSanitizer

```bash
# TSan needs to disable ASLR; add these flags to `docker run` or it exits
# with "FATAL: ThreadSanitizer: encountered an incompatible memory layout"
docker run --rm --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
    -v $(pwd):/workspace ai-cpp-course:local bash

cd /workspace
mkdir -p build-l15 && cd build-l15
cmake ../ai-cpp-l15 -DENABLE_TSAN=ON -DCMAKE_BUILD_TYPE=Debug \
    -Dnanobind_DIR=/usr/local/nanobind/cmake
make tsan_race_demo
./tsan_race_demo correct   # correct ring: PASS, no TSan warning
./tsan_race_demo broken    # WARNING: ThreadSanitizer: data race ...
```

TSan cannot be combined with ASan/UBSan in the same binary (mirrors
[L11](../ai-cpp-l11/)'s `asan_example` and [L20](../ai-cpp-l20/)'s
`ubsan_example`, each their own sanitizer option).

## Build and Run

```bash
cd /workspace
colcon build --packages-select nanobind-l15
source install/setup.bash

pytest ai-cpp-l15/ -v
```

## What You Learned

- A data race is undefined behaviour regardless of whether it ever produces
  a visibly wrong answer on your machine
- The publish sequence matters: write the data, *then* publish the index —
  reversing it is the bug, not the choice of memory order
- `volatile` gives no inter-thread guarantees; `std::atomic` with an
  explicit `memory_order` does
- A lock-free structure's correctness is tied to its concurrency shape
  (SPSC, MPMC, …) — reusing it outside that shape reintroduces races
- `nb::gil_scoped_release` is required for a C++ thread to make real
  progress concurrently with Python, not just run in the background
- ThreadSanitizer detects the race deterministically, independent of
  whether the host's memory model happens to reorder visibly

## Exercises

1. **Find the minimal fix**: `BrokenSpscRing` swaps two lines relative to
   `SpscRing`. Fix it without looking, then diff against `spsc_ring.hpp`.

2. **Remove the delay**: delete the `sleep_for` in `BrokenSpscRing::push`
   and rerun `test_integration_concurrency.py`'s reliability test. Does it
   still fail every time? What does that tell you about the race window?

3. **MPMC**: what specifically about `SpscRing` breaks with two producers?
   (Hint: look at what `head_.load(relaxed)` followed by `head_.store()` is
   not — a read-modify-write.)

4. **TSan on the correct ring**: run `tsan_race_demo correct` under TSan and
   confirm it is clean. Now change one `acquire`/`release` to `relaxed` and
   rerun — which one flips it?

## Lesson Files

| File | Description |
|------|-------------|
| [spsc_ring.hpp](spsc_ring.hpp) | `SpscRing` and `BrokenSpscRing` |
| [concurrency_native.cpp](concurrency_native.cpp) | nanobind module: ring round-trip, GIL release demo |
| [tsan_race_demo.cpp](tsan_race_demo.cpp) | Standalone executable for `-DENABLE_TSAN=ON` |
| [CMakeLists.txt](CMakeLists.txt) | CMake build configuration |
| [test_concurrency.py](test_concurrency.py) | Unit tests: ring correctness, GIL release |
| [test_integration_concurrency.py](test_integration_concurrency.py) | Sustained load, concurrent Python + C++ work |
