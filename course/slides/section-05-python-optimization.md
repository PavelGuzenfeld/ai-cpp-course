# Section 5: Python Optimization for RAM and CPU

## Video 5.1: __slots__ -- Eliminate Per-Instance Overhead (~10 min)

### Slides
- Slide 1: The hidden cost of Python objects -- Every regular Python object carries a `__dict__` (a hash table). For a BoundingBox with 4 fields: object header 48 bytes + __dict__ 120 bytes = ~168 bytes total.
- Slide 2: Adding __slots__ -- `__slots__ = ('x', 'y', 'w', 'h')` eliminates __dict__ entirely. Object size drops to ~72 bytes -- a 40-56% reduction.
- Slide 3: Measuring with tracemalloc -- 100k BoundingBox instances: without slots ~16 MB, with slots ~6.4 MB. The difference (~9.6 MB) is enough to fit in L3 cache instead of spilling to RAM.
- Slide 4: __slots__ + @dataclass (Python 3.10+) -- `@dataclass(slots=True)` auto-generates __slots__. Same memory savings, less boilerplate.
- Slide 5: When NOT to use __slots__ -- Dynamic attribute assignment is impossible. Inheritance requires slots on all parent classes. Rarely an issue for data objects like BoundingBox.

### Key Takeaway
- __slots__ is the single highest-impact Python optimization for objects created in large quantities -- 2x less memory with zero code complexity.

## Video 5.2: Numpy Views vs Copies (~10 min)

### Slides
- Slide 1: The tracker_engine problem -- `HistoryNP.latest()` calls `.copy()` every time. At 30 fps with 100-element buffers, that is 3000 unnecessary allocations per second.
- Slide 2: Views are zero-cost slicing -- `view = buf[90:100]` shares the same underlying memory. `np.shares_memory(buf, view)` returns True. No copy, no allocation.
- Slide 3: When copies are necessary -- The producer mutates the buffer (circular buffer overwrite), the consumer holds the reference long-term (prevents GC of parent), you need contiguous memory for FFI (non-contiguous strides).
- Slide 4: Rule of thumb -- Return a view when the caller uses it immediately (same frame). Return a copy when the caller stores it across frames.
- Slide 5: memoryview and the buffer protocol -- `memoryview` provides a zero-copy window into any object supporting the buffer protocol. The tracker_engine pattern `.clone().cpu().numpy().tolist()` creates 4 allocations for 2 floats -- use `float(arr[0])` instead.

### Key Takeaway
- Numpy views avoid allocation entirely -- use them for immediate consumption within the same frame, and copies only for data that must outlive its source.

## Video 5.3: Pre-Allocated Buffers and Thread Pools (~12 min)

### Slides
- Slide 1: The allocation tax -- tracker_engine's `is_target_moving()` creates 7+ temporary arrays every call. At 30 fps, that is 210+ allocations/second for one method.
- Slide 2: Pre-allocate and reuse pattern -- Allocate in `__init__`, pass `out=` to numpy operations, slice pre-allocated arrays to fit current data size. Zero allocations per frame.
- Slide 3: The thread-per-task problem -- tracker_engine spawns a new `Thread()` for every image save. At 30 fps, 30 threads/second. Thread creation costs ~1 ms. Threads accumulate if I/O is slow.
- Slide 4: ThreadPoolExecutor solution -- Bounded threads (max_workers=4), thread reuse (no creation overhead after warmup), backpressure (busy workers queue tasks), clean shutdown.
- Slide 5: torch.compile overview -- Traces the model's forward pass and generates fused kernels. Operator fusion (4 kernel launches become 1), memory elimination (intermediates never materialized), CUDA graph capture. Modes: default, reduce-overhead, max-autotune.

### Key Takeaway
- Pre-allocated buffers reduce GC pressure in hot loops, and ThreadPoolExecutor bounds resource usage -- both are essential patterns for sustaining 30+ fps.

## Video 5.4: Summary and Before-After Comparison (~8 min)

### Slides
- Slide 1: Before/after table -- __slots__ (~2x less RAM), numpy views (0 allocations), pre-allocated buffers (3-5x faster), memoryview (4x fewer allocs), thread pool (bounded resources), torch.compile (1.5-3x faster).
- Slide 2: The key principle -- Before reaching for C++, squeeze every drop of performance from Python itself. These optimizations have zero external dependencies and work in any Python project.
- Slide 3: Jetson note -- On Jetson, RAM is shared between CPU and GPU (unified memory). Reducing Python memory usage directly frees up memory for GPU operations. __slots__ and pre-allocated buffers are even more impactful on memory-constrained Jetson devices.
- Slide 4: Exercise preview -- Measure your own objects, analyze view safety, measure GC pressure with gc.get_stats(), tune thread pool sizing, torch.compile challenge.

### Key Takeaway
- Always benchmark your specific workload -- not all optimizations help everywhere, but __slots__ and pre-allocation are almost universally beneficial.
