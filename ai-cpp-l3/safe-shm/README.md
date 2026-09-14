# safe-shm

Lock-free shared memory primitives for real-time C++ and Python.

Header-only C++23 library providing seqlock, cyclic ring buffer, time-series temporal queries, and lifecycle management — all over POSIX shared memory with zero-copy cross-process communication. Uses Linux futex for blocking waits, `std::atomic_ref` for lock-free access, and per-slot seqlocks to eliminate contention.

Works on x86_64 and ARM64 Linux.

## Dependencies

- [`shm`](https://github.com/PavelGuzenfeld/shm) — POSIX shared memory wrapper
- [`exception-rt`](https://github.com/PavelGuzenfeld/exception-rt) — runtime exception utilities
- [`fmt`](https://github.com/fmtlib/fmt) — formatting
- [`nanobind`](https://github.com/wjakob/nanobind) (optional) — Python bindings
- [`doctest`](https://github.com/doctest/doctest) (test only) — fetched automatically
- [`nanobench`](https://github.com/martinus/nanobench) (bench only) — fetched automatically

## Components

### Core Primitives

| Header | Class | Description |
|--------|-------|-------------|
| `flat_type.hpp` | `FlatType` | Concept: `trivially_copyable && standard_layout` |
| `seqlock.hpp` | `SeqlockWriter<T>`, `SeqlockReader<T>` | Lock-free single-value read/write with torn-read detection |
| `cyclic_buffer.hpp` | `CyclicBufferWriter<T,N>`, `CyclicBufferReader<T,N>` | Lock-free ring buffer with per-slot seqlock |
| `time_series.hpp` | `TimeSeries<T,N>` | Temporal queries on cyclic buffer (closest, interpolation, freshness) |
| `stamped.hpp` | `Stamped<T>` | Metadata envelope: `{timestamp_ns, sequence, data}` |
| `sanitized_key.hpp` | `SanitizedKey<Tag>` | Strong-typed temporal key with external validator (opt-in) |
| `shm_lifecycle.hpp` | `OwnedShm<T>`, `ShmHeader` | RAII ownership, heartbeat liveness, cleanup utilities |

### Legacy / Specialized

| Header | Class | Description |
|--------|-------|-------------|
| `storage.hpp` | `Storage<T>` | Write-side shared memory (futex-locked) |
| `dblbuf_loader.hpp` | `DblBufLoader<T>` | Double-buffered async reader with background `jthread` |
| `image_shm.hpp` | `DoubleBufferShm<T>` | Combined reader/writer with double buffering |
| `image.hpp` | `Image<W,H,TYPE>` | Compile-time image types (FHD/4K, RGB/RGBA/NV12) |
| `shared_memory.hpp` | `SharedMemory<T>` | Simple typed shm wrapper (no sync) |
| `producer_consumer.hpp` | `ProducerConsumer<T>` | Semaphore-based producer/consumer |
| `shm_stats.hpp` | `ShmStats` | Diagnostic counters with zero-cost opt-out via `[[no_unique_address]]` |

## Architecture

### Seqlock (single value, lock-free)

```
Writer                                  Reader
┌────────────────────┐                 ┌────────────────────┐
│ seq++ (odd)        │                 │ read seq (acquire)  │
│ memcpy data        │  /dev/shm/name  │ memcpy data         │
│ seq++ (even)       ├────────────────►│ read seq (acquire)  │
│ futex_wake         │                 │ if changed → retry  │
└────────────────────┘                 └────────────────────┘
```

### CyclicBuffer (ring buffer, per-slot seqlock)

```
Writer                                     Reader
┌──────────────────────┐                  ┌──────────────────────────┐
│ idx = writes & (N-1) │                  │ get_latest()             │
│ slot[idx].seq++ (odd)│  /dev/shm/name   │ get(reverse_index)       │
│ memcpy slot[idx].data├─────────────────►│ try_get() → optional<T>  │
│ slot[idx].seq++ (even│                  │ wait_for_write() [futex] │
│ total_writes++       │                  │                          │
│ futex_wake           │                  │ TimeSeries layer:        │
└──────────────────────┘                  │  find_closest(key)       │
                                          │  find_interpolation_pair │
  SHM layout:                             │  get_latest_if_fresh     │
  [total_writes] [slot₀] [slot₁] .. [slotₙ₋₁]                      │
  Each slot: [seq_counter | T data]       └──────────────────────────┘
```

### Lifecycle Management

```
OwnedShm<T>
┌──────────────────────────────────┐
│ ShmHeader (32 bytes)             │
│   magic: 0x53484D48             │
│   version, heartbeat_ns         │
│   writer_pid                     │
├──────────────────────────────────┤
│ T data (user payload)            │
└──────────────────────────────────┘
  RAII: unlinks /dev/shm on destruction
  is_writer_alive(): PID check + heartbeat
```

## Choosing a Transport

| Need | Use | Latency |
|------|-----|---------|
| Single latest value, maximum speed | `Seqlock<T>` | ~140 ns (64 B) |
| Ring buffer history + temporal queries | `CyclicBuffer<T,N>` + `TimeSeries` | ~45 ns lookup |
| Double-buffered async loading (images) | `DoubleBufferShm<T>` | ~1 µs (FHD) |
| SHM with ownership tracking | `OwnedShm<T>` | — |
| Simple typed shm (no sync) | `SharedMemory<T>` | ~1 ns |

## Quick Start

### Seqlock (cross-process, lock-free)

```cpp
#include "safe-shm/seqlock.hpp"

// Writer process
safe_shm::SeqlockWriter<SensorData> writer("sensor_shm");
writer.store(SensorData{20.5, 1013.25, 45.0, now_ms()});

// Reader process
safe_shm::SeqlockReader<SensorData> reader("sensor_shm");
auto data = reader.load();                              // non-blocking
auto data2 = reader.load_blocking(last_seq, timeout);   // blocks until new data
```

### CyclicBuffer + TimeSeries (ring buffer with temporal queries)

```cpp
#include "safe-shm/cyclic_buffer.hpp"
#include "safe-shm/stamped.hpp"
#include "safe-shm/time_series.hpp"

using StampedIMU = safe_shm::Stamped<IMUReading>;

// Writer: insert stamped readings at 200 Hz
safe_shm::CyclicBufferWriter<StampedIMU, 64> writer("imu_shm");
writer.insert(safe_shm::stamp(imu_reading, seq++));

// Reader: temporal queries
safe_shm::TimeSeries<StampedIMU, 64> ts("imu_shm");

auto latest = ts.get_latest();
auto closest = ts.find_closest(target_ns);           // binary search O(log N)
auto closest = ts.find_closest(target_ns, 100'000);  // with max_distance guard
auto interp = ts.find_interpolation_pair(target_ns);  // {before, after, alpha}
auto fresh = ts.get_latest_if_fresh(min_timestamp);    // staleness check
```

### Lifecycle Management

```cpp
#include "safe-shm/shm_lifecycle.hpp"

// RAII-managed SHM segment with header
safe_shm::OwnedShm<Config> owned("my_config");
owned.data() = {1.5, 0.0, 3};
owned.update_heartbeat();

// Remote liveness check
if (safe_shm::is_writer_alive(header, /*max_stale_ns=*/1'000'000'000))
    fmt::print("Writer is alive\n");

// Cleanup utilities
for (auto const& seg : safe_shm::shm_list("stale_"))
    safe_shm::shm_remove(seg);
```

### Python Bindings

```python
import safe_shm_py as shm

# Seqlock
writer = shm.SeqlockWriterF64("my_shm")
writer.store(3.14159)

reader = shm.SeqlockReaderF64("my_shm")
value = reader.load()  # 3.14159

# CyclicBuffer + TimeSeries
cb_writer = shm.CyclicBufferWriterStampedF64("ring_shm")
s = shm.StampedF64()
s.timestamp_ns = shm.monotonic_now_ns()
s.sequence = 0
s.data = 42.0
cb_writer.insert(s)

ts = shm.TimeSeriesStampedF64("ring_shm")
closest = ts.find_closest(target_ns)
interp = ts.find_interpolation_pair(target_ns)
```

## Temporal Query Safety

Two layers of protection against unit/clock-type mismatches:

### Layer 1: Runtime — `max_distance` parameter

```cpp
// BUG: querying with milliseconds (5000) instead of nanoseconds
auto bad = ts.find_closest(5000);               // silently returns oldest element
auto bad = ts.find_closest(5000, 100'000'000);  // returns nullopt — mismatch caught!

// CORRECT: consistent nanosecond units
auto good = ts.find_closest(5'000'000'000, 100'000'000);  // works, within 100ms
```

### Layer 2: Compile-time — `SanitizedKey<Tag>` (opt-in)

`SanitizedKey<Tag>` is a strong type that can only be created through an external validator function. The compiler prevents passing raw `uint64_t` where `SanitizedKey` is expected.

```cpp
#include "safe-shm/sanitized_key.hpp"

// Step 1: Sanitize — the ONLY way to create a SanitizedKey
auto key = safe_shm::sanitize<safe_shm::MonotonicNsTag>(
    raw_ns, safe_shm::MonotonicNsValidator{});

if (!key) { /* rejected: wrong units or clock type */ }

// Step 2: Pass to TimeSeries — type-safe overload
auto result = ts.find_closest(*key);
auto interp = ts.find_interpolation_pair(*key);
auto fresh  = ts.get_latest_if_fresh(*key);
```

Built-in validators:

| Validator | Catches |
|-----------|---------|
| `MonotonicNsValidator{}` | Values that aren't plausible `CLOCK_MONOTONIC` nanoseconds |
| `RangeValidator{min, max}` | Values outside expected range |
| `ProximityValidator{ref, tol}` | Values differing from reference by > tolerance |
| `AllOf(v1, v2, ...)` | Composes validators with AND logic |

Different `Tag` types make different sanitization policies type-incompatible at compile time — `SanitizedKey<MonotonicNsTag>` cannot be passed where `SanitizedKey<RangeCheckedTag>` is expected.

Custom validators are just callables:

```cpp
auto key = safe_shm::sanitize<MyTag>(raw_ns, [latest_ts](uint64_t v) {
    return (v > latest_ts ? v - latest_ts : latest_ts - v) <= 2'000'000'000ULL;
});
```

The raw `uint64_t` API is unchanged — `SanitizedKey` is purely opt-in.

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest
```

### Sanitizers

```bash
cmake .. -DSANITIZER=asan    # Address + UB (default)
cmake .. -DSANITIZER=tsan    # Thread sanitizer
cmake .. -DSANITIZER=none    # No sanitizer
```

### Fuzz Testing

```bash
# Requires clang
CC=clang CXX=clang++ cmake .. -DSANITIZER=fuzzer
make fuzz_cyclic_buffer fuzz_time_series
./fuzz_cyclic_buffer corpus_cyclic/ -max_total_time=60
./fuzz_time_series corpus_ts/ -max_total_time=60
```

### Docker (CI)

```bash
docker build -t safe-shm-test -f Dockerfile.test .
docker run --rm -v $(pwd):/src safe-shm-test bash -c \
  "cd /tmp && cmake /src -DSANITIZER=none && make -j\$(nproc) && ctest"
```

## Examples

All examples are buildable targets — see `examples/` directory.

| Example | Language | What it demonstrates |
|---------|----------|---------------------|
| `seqlock_example` | C++ | Cross-process writer/reader with blocking reads |
| `cyclic_buffer_example` | C++ | Ring buffer + Stamped<T> + TimeSeries temporal queries |
| `lifecycle_example` | C++ | OwnedShm RAII, heartbeat, cleanup utilities |
| `python_seqlock.py` | Python | Seqlock read/write + Stamped + monotonic clock |
| `python_cyclic_buffer.py` | Python | CyclicBuffer + TimeSeries from Python |

```bash
# C++ examples (two terminals)
./seqlock_example writer    # terminal 1
./seqlock_example reader    # terminal 2

# Python examples
PYTHONPATH=build python3 examples/python_seqlock.py
PYTHONPATH=build python3 examples/python_cyclic_buffer.py
```

## Tests

| Test | Type | Assertions | What it covers |
|------|------|-----------|---------------|
| `flat_type_test` | Unit | — | FlatType concept acceptance/rejection |
| `double_buffer_swapper_test` | Unit | — | Swap mechanics, structs, large arrays |
| `swap_runner_test` | Unit | — | Trigger/wait, exceptions, concurrent threads |
| `safe_shm_test` | Integration | — | Storage + DblBufLoader round-trip |
| `shared_memory_test` | Integration | — | SharedMemory typed wrapper |
| `image_shm_test` | Integration | — | DoubleBufferShm with FHD images |
| `integration_test` | Integration | — | Cross-process (fork) all components |
| `seqlock_test` | Integration | — | Seqlock store/load, cross-process, concurrent, blocking |
| `stamped_test` | Integration | 28 | Stamped<T> layout, stamp(), monotonic clock, ShmStats |
| `cyclic_buffer_test` | Integration | 24,252 | Insert/get, overflow, structs, cross-process, concurrent |
| `time_series_test` | Integration | 66 | Binary search, interpolation, freshness, edge cases, unit confusion |
| `shm_lifecycle_test` | Integration | 28 | OwnedShm, ShmHeader, heartbeat, liveness detection |
| `cross_process_test` | Integration | 69 | All producer/consumer combos, 1-writer-3-readers, FHD images |
| `sanitized_key_test` | Integration | 41 | SanitizedKey construction, validators, tag safety, TimeSeries integration |

## Benchmarks

Run with: `./benchmark && ./benchmark_competitors` (built with `-O3 -DNDEBUG`, [nanobench](https://github.com/martinus/nanobench))

### Throughput (store + load cycle)

| Method | 64 B | 4 KB | 1 MB | FHD RGB (6 MB) |
|--------|-----:|-----:|-----:|---------------:|
| memcpy (baseline) | 55-60 GB/s | 120-125 GB/s | 26-29 GB/s | 36 GB/s |
| raw POSIX shm (no sync) | 63 GB/s | 128 GB/s | 28 GB/s | 35 GB/s |
| **Seqlock (lock-free)** | **0.4 GB/s** | **9.3 GB/s** | **9.2 GB/s** | **38 GB/s** |
| **CyclicBuffer (per-slot seqlock)** | **0.4 GB/s** | **11.5 GB/s** | **9.1 GB/s** | **39 GB/s** |
| DoubleBufferShm (futex) | ~6 MB/s | 0.15 GB/s | 6.8 GB/s | 12 GB/s |
| POSIX mq (kernel IPC) | 0.16 GB/s | 7.4 GB/s | — | — |
| Unix domain socket | 0.13 GB/s | 6.2 GB/s | — | — |
| pipe (kernel IPC) | 0.17 GB/s | 9.4 GB/s | 8.4 GB/s | — |

### TimeSeries Query Latency

| Operation | Latency | Throughput |
|-----------|--------:|----------:|
| `find_closest` (binary search, 64 elements) | 45 ns | 22 Mop/s |
| `find_interpolation_pair` | 66 ns | 15 Mop/s |
| `get_latest_if_fresh` | 11 ns | 91 Mop/s |

### Key Takeaways

- **CyclicBuffer at FHD**: 39 GB/s — matches raw memcpy with full ring buffer history
- **Seqlock vs CyclicBuffer**: equivalent at small payloads, CyclicBuffer slightly faster at 4 KB
- **vs kernel IPC**: 2.7x faster than pipe at 64 B, 60x faster than POSIX mq at 4 KB
- **TimeSeries lookups**: 45 ns per binary search across 64 elements — suitable for real-time control loops
- **Zero-copy**: no intermediate buffers, no kernel transitions, no thread wakeups on read path

## Design Decisions

**Per-slot seqlock** — Each CyclicBuffer slot has its own 32-bit sequence counter. Readers on slot `i` never contend with writers on slot `j`. The alternative (single global seqlock) forces retry on any concurrent write.

**Power-of-two capacity** — `N & (N-1) == 0` enables bitmask modulo (`idx & (N-1)`) instead of expensive integer division. Enforced at compile time via `static_assert`.

**`Stamped<T>` composability** — `Stamped<T>` is itself a `FlatType`, so `Seqlock<Stamped<SensorData>>`, `CyclicBuffer<Stamped<IMU>, 64>`, and `OwnedShm<Stamped<Config>>` all work without adapter code.

**`std::optional` return types** — `try_get()`, `find_closest()`, `find_interpolation_pair()` return `std::optional<T>` instead of throwing. Compatible with `std::expected` patterns and real-time constraints (no heap allocation).

**Futex for blocking** — `wait_for_write()` and `load_blocking()` use raw `FUTEX_WAIT`/`FUTEX_WAKE` (not `_PRIVATE`) for cross-process signaling. Uncontended path is a single atomic compare — no syscall.

## License

MIT
