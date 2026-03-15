# Section 3 Quiz: Shared Memory, IPC, and Real-Time Patterns

## Q1: Why is shared memory the fastest IPC mechanism?

- a) It uses dedicated hardware acceleration
- b) It compresses data before transmission
- c) Both processes map the same physical RAM into their address spaces, so reading shared data requires no kernel calls, serialization, or copying
- d) It automatically uses the GPU for data transfer

**Answer: c)** Shared memory maps identical physical RAM pages into each process's virtual address space. A read from shared memory is as fast as a read from local memory -- approximately 0.1 microseconds versus 50-100 microseconds for sockets.

## Q2: Why can you NOT safely place a `std::string` into POSIX shared memory?

- a) `std::string` is too large for shared memory
- b) `std::string` contains an internal pointer to heap-allocated character data, which is only valid in the process that created it
- c) `std::string` requires UTF-8 encoding which shared memory does not support
- d) POSIX shared memory only supports integer types

**Answer: b)** A `std::string` stores a pointer to a heap buffer. When another process reads that pointer from shared memory, the address points to the first process's private heap -- not to valid memory in the second process.

## Q3: What does the `FlatType` concept enforce at compile time?

- a) That the type uses less than 64 bytes of memory
- b) That the type is trivially copyable and has standard layout, making it safe for memcpy, shared memory, and GPU transfer
- c) That the type has no member functions
- d) That the type is a primitive integer or float

**Answer: b)** `FlatType` requires `std::is_trivially_copyable_v<T>` and `std::is_standard_layout_v<T>`. This guarantees the type can be safely copied byte-for-byte across process boundaries without pointer fixup or constructor side effects.

## Q4: In a seqlock, how does the reader know if it received valid (non-torn) data?

- a) It checks a CRC checksum appended to the data
- b) It reads the sequence counter before and after reading the data; if both values match and are even, the read was not interrupted by a write
- c) It acquires a mutex lock before reading
- d) It sends an acknowledgment to the writer and waits for confirmation

**Answer: b)** The writer increments the sequence counter to an odd number before writing, then to an even number after. If the reader sees mismatched or odd sequence values, it knows a write was in progress and retries.

## Q5: On a Jetson or similar embedded platform running a multi-process tracking pipeline (camera capture, inference, display), which IPC approach minimizes latency for sharing camera frames between processes?

- a) Writing frames to disk and reading them back
- b) Sending frames over TCP sockets
- c) Using a double-buffered shared memory region with atomic swap
- d) Converting frames to JSON and passing them through a message queue

**Answer: c)** Double-buffered shared memory provides near-zero latency for large image data. The producer writes to the staging buffer while consumers read the active buffer, and an atomic pointer swap makes the new frame visible. This avoids PCIe/network overhead and kernel-mediated copies that embedded systems with tight power and latency budgets cannot afford.

## Q6: What problem does the `exception-rt` submodule solve?

- a) It makes Python exceptions faster
- b) It replaces the default C++ exception allocator with a pre-allocated pool so that `throw` does not call `malloc`, which could block in real-time systems
- c) It converts C++ exceptions to error codes
- d) It logs all exceptions to a file

**Answer: b)** In real-time systems, `throw` can trigger `malloc`, which acquires a lock and may block for an unbounded duration. `exception-rt` provides a static pool of pre-allocated exception slots, making `throw` an O(1) operation with deterministic timing.

## Q7: What advantage does nanobind have over pybind11 for numpy interop?

- a) nanobind supports Python 2 while pybind11 does not
- b) nanobind's `nb::ndarray` enables zero-copy views into C++ memory, avoiding the memcpy overhead of pybind11's `py::array_t`
- c) nanobind compiles C++ code into pure Python
- d) nanobind automatically parallelizes numpy operations

**Answer: b)** nanobind's `nb::ndarray` can return numpy arrays backed directly by C++ memory with no data copy, while pybind11's `py::array_t` typically involves copying data between C++ and Python memory regions.

## Q8: Why does the cyclic buffer use a power-of-2 capacity?

- a) It looks cleaner in code
- b) Power-of-2 sizes allow modulo indexing to be computed with a fast bitwise AND instead of an expensive division operation
- c) Memory allocators only support power-of-2 sizes
- d) It ensures the buffer fits exactly in a cache line

**Answer: b)** Computing `index % capacity` requires integer division, which takes 20-40 CPU cycles. When capacity is a power of 2, the same result is achieved with `index & (capacity - 1)`, which takes a single cycle.
