# Section 5 Quiz: Python Optimization for RAM and CPU

## Q1: What does adding `__slots__` to a Python class eliminate?

- a) The class's `__init__` method
- b) The per-instance `__dict__` hash table, reducing memory usage by roughly 40-56% per object
- c) The ability to define methods on the class
- d) The class's inheritance chain

**Answer: b)** A regular Python object carries a `__dict__` (a hash table costing ~120 bytes). `__slots__` replaces it with a fixed-size struct of named attribute slots, eliminating the dictionary overhead entirely. For small objects created in large quantities, this can halve memory usage.

## Q2: When is it safe to return a numpy view instead of a copy from a circular buffer?

- a) Always -- views are always safe
- b) When the caller will use the data immediately within the same frame, before the buffer is overwritten
- c) Only when the data is less than 1 KB
- d) Only when the buffer uses float64 dtype

**Answer: b)** A view shares memory with the source buffer. If the buffer is overwritten (e.g., the next frame's data overwrites the region), the view sees corrupted data. Views are safe when the caller consumes the data before the producer writes again.

## Q3: A function that creates 7 temporary numpy arrays per call at 30 FPS generates how many allocations per second?

- a) 7
- b) 30
- c) 210
- d) 2100

**Answer: c)** 7 arrays per call multiplied by 30 calls per second = 210 allocations per second. Each allocation involves malloc, array initialization, and eventual garbage collection.

## Q4: What is the primary advantage of using `np.subtract(a, b, out=result)` over `result = a - b`?

- a) It produces more accurate floating-point results
- b) It reuses the pre-allocated `result` array instead of allocating a new one, eliminating allocation overhead
- c) It runs the subtraction on the GPU
- d) It automatically parallelizes the operation across cores

**Answer: b)** The `out=` parameter tells numpy to write results into an existing array. This avoids malloc, array initialization, and garbage collection overhead that `result = a - b` incurs by creating a new array.

## Q5: Why is `ThreadPoolExecutor` preferred over spawning a new `Thread()` for each I/O task?

- a) `ThreadPoolExecutor` supports GPU operations while `Thread` does not
- b) `ThreadPoolExecutor` reuses a bounded number of threads, avoiding the overhead of creating ~30 threads/second and preventing unbounded resource consumption
- c) `ThreadPoolExecutor` bypasses the GIL
- d) `ThreadPoolExecutor` uses faster system calls

**Answer: b)** Creating a thread costs ~1 ms on Linux. At 30 FPS, spawning a new thread per frame creates 30 threads/second that may accumulate if I/O is slow, eventually exhausting OS resources. A pool reuses a fixed set of threads and queues excess work.

## Q6: What does `torch.compile(mode="reduce-overhead")` optimize beyond the default mode?

- a) It reduces the file size of the model
- b) It captures CUDA graphs to eliminate kernel launch overhead entirely, in addition to operator fusion
- c) It reduces the number of model parameters
- d) It switches from float32 to int8 quantization

**Answer: b)** The `reduce-overhead` mode performs CUDA graph capture on top of the standard operator fusion. This eliminates the CPU-side overhead of launching individual GPU kernels, which is particularly beneficial for inference loops with many small operations.

## Q7: Converting a tensor with `.clone().cpu().numpy().tolist()` to get two float values involves how many unnecessary memory operations?

- a) 0 -- this is the optimal approach
- b) 1
- c) 3 unnecessary copies/allocations when `float(tensor[0])` would suffice
- d) 5

**Answer: c)** `.clone()` copies GPU memory, `.cpu()` transfers to CPU, `.numpy()` creates a numpy view (cheap), and `.tolist()` allocates Python objects. For extracting two floats, `float(tensor[0])` and `float(tensor[1])` or `.item()` avoids all intermediate copies.

## Q8: How does Python 3.10's `@dataclass(slots=True)` compare to manually writing `__slots__`?

- a) It provides the same memory savings with less boilerplate -- `__slots__` is auto-generated from the field annotations
- b) It is slower because dataclasses add overhead
- c) It only works for immutable classes
- d) It requires a third-party library

**Answer: a)** The `slots=True` parameter tells the dataclass decorator to automatically generate `__slots__` from the declared fields. The memory savings are identical to manual `__slots__`, but you avoid writing the redundant tuple of attribute names.
