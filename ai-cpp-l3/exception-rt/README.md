# exception-rt

Drop-in C++ exception allocation override using static storage. Link this library and all exception allocations become deterministic — no `malloc`, no `mmap`, no syscalls.

## How it works

The C++ runtime uses two weak symbols for exception object allocation:
- `__cxa_allocate_exception` — allocates memory for a thrown object
- `__cxa_free_exception` — frees it after the catch block

This library provides strong definitions that override them at link time. Instead of heap allocation, exceptions are served from a fixed pool of pre-allocated slots in static memory.

## Usage

```cmake
find_package(exception-rt REQUIRED)
target_link_libraries(my_target PRIVATE exception-rt::exception-rt)
```

That's it. Any `throw` in code linked against this library will use static allocation.

## Configuration

Override defaults via compile defines:

| Define | Default | Description |
|--------|---------|-------------|
| `EXCEPTION_RT_ABI_HEADER_SIZE` | 128 | Space reserved for the C++ ABI header (`__cxa_refcounted_exception`) |
| `EXCEPTION_RT_MAX_THROWN_SIZE` | 256 | Maximum size of a thrown object in bytes |
| `EXCEPTION_RT_NUM_SLOTS` | 8 | Number of concurrent in-flight exceptions |

Total static memory: `NUM_SLOTS * (ABI_HEADER_SIZE + MAX_THROWN_SIZE)` = 3072 bytes by default.

```cmake
target_compile_definitions(my_target PRIVATE
    EXCEPTION_RT_NUM_SLOTS=16
    EXCEPTION_RT_MAX_THROWN_SIZE=512
)
```

## Introspection API

```cpp
#include <exception-rt/exception.hpp>

exception_rt::slots_in_use();     // currently allocated slots
exception_rt::num_slots();        // compile-time pool size
exception_rt::max_thrown_size();   // compile-time max thrown object size
exception_rt::owns(ptr);          // true if ptr is within the pool
```

## Failure modes

| Condition | Behavior |
|-----------|----------|
| Thrown object exceeds `MAX_THROWN_SIZE` | `__builtin_trap()` with message to stderr |
| All slots exhausted | `__builtin_trap()` with message to stderr |
| Freeing unknown pointer | `__builtin_trap()` with message to stderr |

All failures are loud and immediate. No silent corruption.

## Thread safety

Slot allocation is lock-free using `std::atomic<bool>` with acquire/release ordering. Safe for concurrent throws from multiple threads.

## Limitations

- **Itanium ABI only** — works with GCC and Clang on Linux. Not compatible with MSVC.
- **ABI header size is platform-dependent** — the default (128) matches `sizeof(__cxa_refcounted_exception)` on x86_64 GCC. The test suite includes a `static_assert` that catches mismatches at compile time.
- **Fixed pool** — if you need more concurrent exceptions than `NUM_SLOTS`, increase it at compile time.

## Building

```bash
cmake -B build
cmake --build build
ctest --test-dir build
```

## License

MIT
