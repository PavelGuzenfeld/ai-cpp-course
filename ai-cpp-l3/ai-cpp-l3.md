# Lesson 3: Shared Memory, IPC, and Real-Time Patterns

## Goal

Move beyond single-process Python↔C++ bindings into inter-process
communication (IPC). Learn how to share images and data between processes
using POSIX shared memory, and explore lock-free patterns for real-time
systems where latency matters more than throughput.

## Why Shared Memory?

In a production tracking system, multiple processes run simultaneously:
- **Camera capture** (C++) — reads frames from hardware
- **Tracker inference** (Python) — runs the neural network
- **Display/telemetry** (C++) — renders overlays, sends data to GCS

These processes cannot share memory through function calls. Options:

| Method | Latency | Bandwidth | Complexity |
|--------|---------|-----------|------------|
| Sockets (TCP/UDP) | ~50-100 µs | ~1 GB/s | Low |
| Pipes/FIFOs | ~10-50 µs | ~2 GB/s | Low |
| Message queues | ~20-100 µs | ~1 GB/s | Medium |
| **Shared memory** | **~0.1 µs** | **Memory speed** | Medium |
| DMA/GPU direct | ~1 µs | ~12 GB/s | High |

Shared memory maps the same physical RAM into multiple processes' address
spaces. Reading shared data is as fast as reading local memory — no kernel
calls, no serialization, no copying.

## POSIX Shared Memory Basics

### The `shm` submodule

```cpp
#include <shm/shm.hpp>

// Create or open a shared memory region
shm::Shm segment("/my_data", sizeof(MyStruct));

// Get a pointer to the shared data
MyStruct* data = static_cast<MyStruct*>(segment.data());

// Write from process A
data->value = 42;

// Read from process B (same /my_data name)
std::cout << data->value;  // 42
```

Under the hood:
1. `shm_open()` — creates a file descriptor in `/dev/shm/`
2. `ftruncate()` — sets the size
3. `mmap()` — maps the file into the process's virtual address space
4. The pointer is now usable like any other memory

### RAII Cleanup

The `shm::Shm` class follows RAII — when it goes out of scope:
- `munmap()` unmaps the memory
- `close()` closes the file descriptor
- Optionally `shm_unlink()` removes the shared segment

This prevents resource leaks even when exceptions occur.

## The FlatType Constraint

**Not all types can go into shared memory.** If you put a `std::string` into
shared memory, the other process reads a dangling pointer:

```
Process A's memory:
  SharedData.name → std::string{ ptr → 0x7f8a0042 "hello" }

Process B sees:
  SharedData.name → std::string{ ptr → 0x7f8a0042 ← INVALID ADDRESS! }
```

The `FlatType` concept prevents this at compile time:

```cpp
template <typename T>
concept FlatType = std::is_trivially_copyable_v<T>
                && std::is_standard_layout_v<T>;
```

A `FlatType` is:
- **Trivially copyable**: Can be copied with `memcpy` (no copy constructor side effects)
- **Standard layout**: Memory layout is predictable (no virtual functions, no base class weirdness)

Types that satisfy `FlatType`: `int`, `float`, `struct { int x; float y; }`,
fixed-size arrays.

Types that do NOT: `std::string`, `std::vector`, anything with a pointer.

## Lock-Free Patterns: The `safe-shm` Library

The `safe-shm` submodule provides production-ready lock-free primitives:

### Seqlock (Sequence Lock)

A seqlock allows one writer and multiple readers without blocking:

```
Writer:                          Reader:
1. sequence++ (now odd = writing)   1. Read sequence (s1)
2. Write data                       2. Read data
3. sequence++ (now even = done)     3. Read sequence (s2)
                                    4. If s1 != s2 or s1 is odd → retry
```

No mutex, no blocking. The reader may retry, but it never waits.

```cpp
#include <safe-shm/seqlock.hpp>

// Writer
SeqlockWriter<MyData> writer("/sensor_data");
writer.write(MyData{42, 3.14});

// Reader (may be in another process)
SeqlockReader<MyData> reader("/sensor_data");
auto data = reader.read();  // Retries automatically if torn
```

### Cyclic Buffer (Lock-Free Ring Buffer)

For streaming data (e.g., position history, sensor readings):

```cpp
CyclicBufferWriter<SensorReading, 1024> writer("/readings");
writer.push(SensorReading{timestamp, value});

CyclicBufferReader<SensorReading, 1024> reader("/readings");
auto latest = reader.latest(10);  // Last 10 readings
```

- Power-of-2 capacity for fast modulo (bitwise AND)
- Per-slot seqlocks prevent torn reads
- Futex-based blocking for efficient waiting

### Double Buffer

For large data like images, where you cannot atomically update all bytes:

```
Buffer A (active):  [Frame N - being read by consumers]
Buffer B (staging): [Frame N+1 - being written by producer]

Swap: atomic pointer exchange → consumers now read Frame N+1
```

```cpp
DoubleBufferShm<Image4K_RGB> shm("/camera");

// Producer
shm.stage(frame);   // Write to staging buffer
shm.swap();         // Atomic swap — consumers see the new frame

// Consumer
auto snapshot = shm.snapshot();  // Get current active buffer
process(*snapshot);              // Zero-copy access to image data
```

## nanobind for Python Bindings

Lesson 3 introduces **nanobind** as the successor to pybind11 (used in L1/L2).
The `nanobind-example/` directory shows the basics:

```cpp
#include <nanobind/nanobind.h>
namespace nb = nanobind;

class MyClass {
public:
    MyClass(int x) : value_{x} {
        if (x < 0) throw std::invalid_argument("must be non-negative");
    }
    int get_value() const noexcept { return value_; }
    void set_value(int x) { /* ... */ }
private:
    int value_;
};

NB_MODULE(my_module, m) {
    nb::class_<MyClass>(m, "MyClass")
        .def(nb::init<int>(), nb::arg("value") = 0)
        .def("get_value", &MyClass::get_value)
        .def("set_value", &MyClass::set_value);
}
```

The `safe-shm` library includes full nanobind bindings for all its primitives,
allowing Python processes to participate in shared-memory IPC:

```python
import safe_shm_py as shm

writer = shm.SeqlockWriter_f64("/test", 0.0)
writer.write(42.0)

reader = shm.SeqlockReader_f64("/test")
value = reader.read()  # 42.0
```

## Deterministic Exception Allocation: `exception-rt`

In real-time systems, `throw` is dangerous because it may call `malloc`
(which can block). The `exception-rt` submodule replaces the default exception
allocator with a static pool:

```cpp
// Before: throw may call malloc → unbounded latency
throw std::runtime_error("sensor timeout");

// After (with exception-rt linked): throw uses a pre-allocated slot → O(1)
throw std::runtime_error("sensor timeout");  // Same syntax, deterministic timing
```

This is invisible to application code — just link `exception-rt` and exceptions
become real-time safe.

## Build and Run

```bash
cd /workspace

# Initialize submodules
git submodule update --init --recursive

# Build all lesson 3 components
colcon build
source install/setup.bash

# Run the nanobind example
python3 ai-cpp-l3/nanobind-example/nanobind_example.py
```

## Exercises

1. **Shared memory round-trip**: Write a C++ program that creates shared
   memory, writes a struct, then write a Python script that reads it using
   the nanobind bindings. Measure the latency.

2. **Seqlock vs mutex**: Benchmark a seqlock reader vs a mutex-protected read
   with one writer and multiple readers. At what reader count does the seqlock
   pull ahead?

3. **Image streaming**: Use `DoubleBufferShm<Image>` to stream synthetic images
   between two processes. Measure frames per second.

4. **FlatType violations**: Try to instantiate `SharedMemory<std::string>`.
   What compiler error do you get? Why is this better than a runtime crash?

## What You Learned

- POSIX shared memory maps the same physical RAM into multiple processes
- `FlatType` concept prevents unsafe types from entering shared memory
- Seqlocks provide wait-free reads for single-writer/multi-reader scenarios
- Double buffering enables zero-copy image sharing between processes
- nanobind is the modern successor to pybind11 (smaller, faster, zero-copy)
- Deterministic exception allocation makes C++ exceptions real-time safe

## Files and Submodules

| File/Directory | Purpose |
|----------------|---------|
| `nanobind-example/` | Basic nanobind binding example |
| `shm/` | POSIX shared memory RAII wrapper |
| `safe-shm/` | Lock-free shared memory library (seqlock, cyclic buffer, double buffer) |
| `exception-rt/` | Deterministic exception allocator for real-time |
| `single-task-runner/` | Thread-based task execution utility |
| `double-buffer-swapper/` | Core double-buffer swap primitive |
| `flat-type/` | FlatType concept definition |
| `image-shm-dblbuf/` | Image-specific shared memory double buffer |

## Dependencies

```bash
sudo apt-get install libfmt-dev
```

## References

- [nanobind documentation](https://github.com/wjakob/nanobind)
- [nanobind example project](https://github.com/wjakob/nanobind_example)
- [POSIX shared memory (man page)](https://man7.org/linux/man-pages/man7/shm_overview.7.html)
