# Section 3: Shared Memory, IPC, and Real-Time Patterns

## Video 3.1: Why Shared Memory? (~10 min)

### Slides
- Slide 1: The multi-process tracking system -- In production, multiple processes run simultaneously: camera capture (C++), tracker inference (Python), display/telemetry (C++). These cannot share memory through function calls.
- Slide 2: IPC methods compared -- Sockets (50-100 us, ~1 GB/s), Pipes (10-50 us, ~2 GB/s), Message queues (20-100 us, ~1 GB/s), Shared memory (0.1 us, memory speed), DMA/GPU direct (1 us, ~12 GB/s). Shared memory is the fastest by far.
- Slide 3: How shared memory works -- `shm_open()` creates a file descriptor in `/dev/shm/`, `ftruncate()` sets the size, `mmap()` maps it into the process's virtual address space. The pointer works like any other memory.
- Slide 4: RAII cleanup -- The `shm::Shm` class follows RAII -- when it goes out of scope, `munmap()` unmaps the memory, `close()` closes the descriptor, optionally `shm_unlink()` removes the segment. No resource leaks even with exceptions.
- Slide 5: Jetson IPC note -- On Jetson, shared memory is particularly useful for camera-to-inference pipelines. The V4L2 camera driver can map frames directly into shared memory, avoiding copies. Combined with CUDA unified memory (covered in L7), the entire pipeline from camera to GPU can be zero-copy.

### Key Takeaway
- Shared memory maps the same physical RAM into multiple processes -- reads are as fast as local memory access with no kernel calls or serialization.

## Video 3.2: The FlatType Constraint (~8 min)

### Slides
- Slide 1: Not all types can go into shared memory -- If you put a `std::string` into shared memory, the other process reads a dangling pointer. The string's internal pointer is valid only in the originating process's address space.
- Slide 2: The FlatType concept -- `concept FlatType = std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>`. A FlatType can be copied with `memcpy`, mapped into shared memory, sent to a GPU, and serialized to disk.
- Slide 3: Valid FlatTypes -- `int`, `float`, `struct { int x; float y; }`, fixed-size arrays. Invalid: `std::string`, `std::vector`, anything with a pointer member.
- Slide 4: Compile-time enforcement -- The concept prevents unsafe types at compile time. `SharedMemory<std::string>` produces a clear error: "constraint not satisfied: FlatType." This is better than a runtime crash.

### Key Takeaway
- The FlatType concept catches memory-unsafe shared memory usage at compile time, turning runtime crashes into compiler errors.

## Video 3.3: Lock-Free Patterns (~12 min)

### Slides
- Slide 1: Seqlock (Sequence Lock) -- One writer, multiple readers, no blocking. Writer increments sequence (odd = writing), writes data, increments again (even = done). Reader checks sequence before and after reading; retries if sequence changed or is odd.
- Slide 2: Seqlock code example -- `SeqlockWriter<MyData>` and `SeqlockReader<MyData>`. The reader's `read()` method retries automatically if a torn read is detected.
- Slide 3: Cyclic buffer (lock-free ring buffer) -- For streaming data like position history or sensor readings. Power-of-2 capacity for fast modulo (bitwise AND). Per-slot seqlocks prevent torn reads. Futex-based blocking for efficient waiting.
- Slide 4: Double buffer for images -- For large data where you cannot atomically update all bytes. Two buffers: active (being read) and staging (being written). Atomic pointer swap makes consumers see the new frame instantly.
- Slide 5: Double buffer code -- `shm.stage(frame)` writes to staging, `shm.swap()` does atomic swap, `shm.snapshot()` gets current active buffer. Zero-copy access to image data.

### Key Takeaway
- Lock-free primitives (seqlock, cyclic buffer, double buffer) provide real-time guarantees -- readers never block, even when the writer is active.

## Video 3.4: Nanobind Introduction and Python Bindings (~10 min)

### Slides
- Slide 1: nanobind vs pybind11 -- nanobind is the successor to pybind11, created by the same author. 3-5x smaller binaries, 2-3x faster compile, zero-copy ndarray, better move semantics.
- Slide 2: Basic nanobind binding -- `NB_MODULE(my_module, m)` with `nb::class_<MyClass>`, `.def(nb::init<int>())`, `.def("method", &MyClass::method)`.
- Slide 3: safe-shm Python bindings -- The safe-shm library includes full nanobind bindings. Python processes can participate in shared-memory IPC: `shm.SeqlockWriter_f64("/test", 0.0)`, `writer.write(42.0)`, `reader.read()`.
- Slide 4: Deterministic exception allocation (exception-rt) -- In real-time systems, `throw` may call `malloc` which can block. The exception-rt submodule replaces the default exception allocator with a static pool. Same syntax, deterministic timing.

### Live Demo
- Run the nanobind example showing Python-to-Python IPC through shared memory. Start a writer in one terminal, a reader in another, demonstrate real-time data sharing.

### Key Takeaway
- nanobind is the modern choice for C++-to-Python bindings -- smaller, faster, and with zero-copy numpy interop that we will use extensively in Lesson 4.
