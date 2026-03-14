# Lesson 5: Python Optimization for RAM & CPU

Before reaching for C++, squeeze every drop of performance from Python itself.
This lesson covers the highest-impact optimizations, all demonstrated with
patterns taken from **tracker_engine** (a real-time UAV tracker written in pure
Python).

## Build and Run

This lesson is pure Python — no C++ build required.

```bash
# Inside Docker container (or any environment with Python 3.10+ and numpy)
cd /workspace/ai-cpp-l5

# Run individual demos
python3 bbox_slots.py
python3 numpy_views.py
python3 preallocated_buffers.py
python3 thread_pool_io.py

# Run full benchmark suite
python3 benchmark_python_opt.py

# Run tests
pytest ai-cpp-l5/ -v
```

---

## Recall: The Memory Hierarchy (from L2)

| Level | Latency | Size |
|-------|---------|------|
| L1 cache | ~0.5 ns | 64 KB |
| L2 cache | ~7 ns | 256 KB--1 MB |
| L3 cache | ~20 ns | 8--32 MB |
| RAM | ~100 ns | 16--64 GB |

Every Python `dict` lookup, every temporary numpy array, every unnecessary copy
pushes hot data out of cache. The optimizations below reduce memory traffic and
keep data where the CPU can reach it fast.

---

## 1. `__slots__` -- Eliminate Per-Instance `__dict__`

Every regular Python object carries a `__dict__` (a hash table). For small
objects created by the millions, this wastes 100+ bytes each.

### Before (tracker_engine style)

```python
class BoundingBox:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    @property
    def area(self):
        return self.w * self.h
```

```python
>>> import sys
>>> b = BoundingBox(10, 20, 100, 50)
>>> sys.getsizeof(b)           # object header
48
>>> sys.getsizeof(b.__dict__)  # the hidden hash table
120  # ~168 bytes total per bbox
```

### After -- with `__slots__`

```python
class BoundingBox:
    __slots__ = ('x', 'y', 'w', 'h')

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    @property
    def area(self):
        return self.w * self.h
```

```python
>>> b = BoundingBox(10, 20, 100, 50)
>>> sys.getsizeof(b)   # no __dict__ at all
72
>>> hasattr(b, '__dict__')
False
```

**Savings**: ~40-56% less memory per object (varies by Python version). With 100k bounding boxes, that is
~9.6 MB saved -- enough to fit in L3 cache instead of spilling to RAM.

### Measuring with `tracemalloc`

```python
import tracemalloc

tracemalloc.start()
boxes = [BoundingBox(i, i, 10, 10) for i in range(100_000)]
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"Current: {current / 1024:.1f} KB, Peak: {peak / 1024:.1f} KB")
```

---

## 2. `__slots__` + `@dataclass` (Python 3.10+)

Python 3.10 added native slots support to dataclasses:

```python
from dataclasses import dataclass

@dataclass(slots=True)
class BoundingBox:
    x: float
    y: float
    w: float
    h: float

    @property
    def area(self):
        return self.w * self.h
```

Same memory savings, less boilerplate. The `slots=True` parameter auto-generates
`__slots__` and prevents `__dict__` creation.

---

## 3. Numpy Views vs. Copies

### The Problem

tracker_engine's `HistoryNP.latest()` always calls `.copy()`:

```python
def latest(self, n=5):
    # ... slicing logic ...
    return self._buffer[start:end].copy()  # always copies!
```

Every call allocates a new array and copies data. If `latest()` is called once
per frame at 30 fps with 100-element buffers, that is 3000 unnecessary
allocations per second.

### Views: Zero-Copy Slicing

Numpy slices return **views** -- they share the same underlying memory:

```python
import numpy as np

buf = np.zeros((100, 4), dtype=np.float32)
view = buf[90:100]        # no copy -- view into buf's memory
copy = buf[90:100].copy() # new allocation

np.shares_memory(buf, view)  # True
np.shares_memory(buf, copy)  # False
```

### When You Need Copies

Views are unsafe when:

1. **The producer mutates the buffer**: if `buf` is a circular buffer that gets
   overwritten, a view into it will see corrupted data.
2. **The consumer holds the reference long-term**: keeping a view alive prevents
   the entire source array from being garbage collected.
3. **You need contiguous memory for FFI**: views from non-contiguous slices
   (e.g., `buf[::2]`) are not contiguous.

**Rule of thumb**: return a view when the caller will use it immediately (same
frame). Return a copy when the caller stores it across frames.

---

## 4. Pre-Allocated Buffers

### The Problem

tracker_engine's `VelocityTracker.is_target_moving()` creates 7+ temporary
arrays every call:

```python
def is_target_moving(self):
    positions = np.array(self.history)      # alloc 1
    diffs = np.diff(positions, axis=0)      # alloc 2
    distances = np.linalg.norm(diffs, ...)  # alloc 3
    weights = np.exp(...)                   # alloc 4
    weighted = distances * weights          # alloc 5
    ema = np.cumsum(weighted)               # alloc 6
    velocity = ema[-1] / np.sum(weights)    # alloc 7
    return velocity > self.threshold
```

At 30 fps, that is 210+ allocations/second just for this one method.

### The Fix: Pre-Allocate and Reuse

```python
class VelocityTracker:
    def __init__(self, history_len=20):
        self._diffs = np.empty((history_len - 1, 2), dtype=np.float64)
        self._distances = np.empty(history_len - 1, dtype=np.float64)
        self._weights = np.empty(history_len - 1, dtype=np.float64)
        self._weighted = np.empty(history_len - 1, dtype=np.float64)

    def is_target_moving(self, positions):
        n = len(positions) - 1
        np.subtract(positions[1:n+1], positions[:n], out=self._diffs[:n])
        np.linalg.norm(self._diffs[:n], axis=1, out=self._distances[:n])  # nope -- no out param
        # Use manual computation instead:
        np.multiply(self._diffs[:n, 0], self._diffs[:n, 0], out=self._weighted[:n])
        np.multiply(self._diffs[:n, 1], self._diffs[:n, 1], out=self._distances[:n])
        np.add(self._weighted[:n], self._distances[:n], out=self._distances[:n])
        np.sqrt(self._distances[:n], out=self._distances[:n])
        # ... etc
```

**Key idea**: allocate once in `__init__`, pass `out=` to numpy operations,
slice pre-allocated arrays to fit the current data size.

---

## 5. `memoryview` and the Buffer Protocol

### The Problem

tracker_engine converts tensors through multiple formats:

```python
shift = full_shift.clone().cpu().numpy().tolist()
# tensor -> tensor copy -> CPU tensor -> numpy array -> Python list
# That's 4 allocations for 2 floats!
```

### The Fix: Use the Buffer Protocol

```python
# numpy and torch share the buffer protocol
arr = full_shift.cpu().numpy()   # shares memory with tensor (when possible)
view = memoryview(arr)           # zero-copy view

# Access individual values without creating a list
x, y = float(arr[0]), float(arr[1])

# Or pass directly to functions that accept buffer protocol objects
```

`memoryview` provides a zero-copy window into any object that supports the
buffer protocol (bytes, bytearray, numpy arrays, etc.):

```python
data = bytearray(1024 * 1024)  # 1 MB
view = memoryview(data)[100:200]  # zero-copy slice
# view uses no additional memory
```

---

## 6. Thread Pools vs. Thread-Per-Task

### The Problem

tracker_engine spawns a new `Thread()` for every image save:

```python
def save_log_image(self, image, path):
    t = threading.Thread(target=cv2.imwrite, args=(path, image))
    t.start()
    # no join, no limit on concurrent threads
```

At 30 fps, this creates 30 threads/second. Thread creation costs ~1 ms on
Linux. Threads accumulate if I/O is slow, eventually exhausting OS resources.

### The Fix: Thread Pool with Bounded Concurrency

```python
from concurrent.futures import ThreadPoolExecutor

class ImageSaver:
    def __init__(self, max_workers=4):
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def save(self, image, path):
        self._pool.submit(cv2.imwrite, path, image)

    def shutdown(self):
        self._pool.shutdown(wait=True)
```

Benefits:
- **Bounded threads**: at most `max_workers` threads exist at any time
- **Thread reuse**: no creation overhead after warmup
- **Backpressure**: when all workers are busy, `submit()` queues the task
- **Clean shutdown**: `shutdown(wait=True)` waits for all pending saves

---

## 7. `torch.compile()`

### The Problem

tracker_engine has `torch.compile()` commented out:

```python
# @torch.compile()
def forward(self, x):
    ...
```

### What `torch.compile()` Does

`torch.compile()` traces your model's forward pass and generates optimized
fused kernels:

```python
import torch

@torch.compile(mode="reduce-overhead")
def forward(x):
    y = torch.relu(x)
    z = y * 2 + 1
    return z.mean()

# First call: traces and compiles (~seconds)
# Subsequent calls: runs compiled code (~microseconds faster)
```

What it optimizes:
1. **Operator fusion**: `relu + mul + add + mean` becomes one kernel launch
   instead of four
2. **Memory elimination**: intermediate tensors (`y`, `z`) are never
   materialized in RAM
3. **CUDA graph capture** (with `mode="reduce-overhead"`): eliminates kernel
   launch overhead entirely

### Modes

| Mode | Compile time | Runtime gain | Use case |
|------|-------------|--------------|----------|
| `default` | Medium | Good | General |
| `reduce-overhead` | High | Best | Inference loops |
| `max-autotune` | Very high | Best possible | Benchmarking |

### Caveats

- First call is slow (compilation)
- Dynamic shapes trigger recompilation
- Not all Python constructs are traceable (data-dependent control flow breaks)
- Use `torch._dynamo.config.suppress_errors = True` during development

---

## Summary: Before & After

| Optimization | Before | After | Typical Gain |
|-------------|--------|-------|--------------|
| `__slots__` | ~152 bytes/bbox | ~64 bytes/bbox | ~2x less RAM |
| numpy views | `.copy()` every call | slice view | 0 allocations |
| Pre-allocated buffers | 7 arrays/frame | 0 arrays/frame | 3--5x faster |
| `memoryview` | 4 conversions | 1 conversion | 4x fewer allocs |
| Thread pool | Thread() per save | Pool of 4 | bounded resources |
| `torch.compile()` | interpreted | fused kernels | 1.5--3x faster |

---

## Exercises

1. **Measure your objects**: Pick a class from your own codebase. Add `__slots__`
   and measure memory savings with `tracemalloc` for 100k instances.

2. **View safety analysis**: In `numpy_views.py`, the wrap-around demo shows
   data corruption from views. Write a function that decides automatically
   whether to return a view or copy based on whether the caller is in the
   "current frame" or "storing across frames."

3. **GC pressure**: Run `gc.get_stats()` before and after 10,000 iterations of
   `VelocityTrackerSlow` vs `VelocityTrackerFast`. How many Gen-0 collections
   does each trigger?

4. **Thread pool sizing**: In `thread_pool_io.py`, vary `max_workers` from 1 to
   32. Plot throughput vs workers. Where is the sweet spot?

5. **torch.compile challenge**: Take a simple PyTorch post-processing function
   (e.g., NMS + score filtering) and compare with/without `@torch.compile`.
   When does compilation overhead outweigh the benefit?

## What You Learned

- `__slots__` eliminates `__dict__` overhead — 2x less memory per object
- Numpy views avoid allocation; copies are needed only for long-lived data
- Pre-allocated buffers reduce GC pressure in hot loops
- `ThreadPoolExecutor` bounds resource usage vs unbounded `Thread()` creation
- `torch.compile()` fuses operations but has compilation overhead
- Always benchmark your specific workload — not all optimizations help everywhere

---

## Files in This Lesson

| File | What It Demonstrates |
|------|---------------------|
| `bbox_slots.py` | `__slots__`, `@dataclass(slots=True)`, memory measurement |
| `numpy_views.py` | Views vs. copies, `np.shares_memory()` |
| `preallocated_buffers.py` | Pre-allocated numpy buffers |
| `thread_pool_io.py` | `ThreadPoolExecutor` vs. `Thread()` per task |
| `benchmark_python_opt.py` | Benchmark all optimizations with timing + memory |
| `test_python_opt.py` | Unit tests for all variants |
| `test_integration_python_opt.py` | Integration test simulating a tracking loop |
