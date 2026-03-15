# Section 7: NVIDIA GPU Programming -- Keeping Data Where It Belongs

## Video 7.1: PCIe Is the Bottleneck (~10 min)

### Slides
- Slide 1: The real problem -- PCIe 3.0 x16 bandwidth ~12 GB/s vs GPU memory bandwidth ~936 GB/s (RTX 3090). GPU memory is 75x faster than the PCIe bus. Every `.cpu()`, `.numpy()`, or `.to(device)` call pays the PCIe tax.
- Slide 2: tracker_engine anti-pattern 1 -- `prepare_boxes()` calls `.cpu()` on boxes and scores before NMS. But `torchvision.ops.nms` supports GPU tensors directly. Two unnecessary PCIe transfers per frame.
- Slide 3: tracker_engine anti-pattern 2 -- `os_tracker_forward()` creates `torch.from_numpy(image).to(device)` every frame. This means CPU allocation, GPU allocation, PCIe transfer, and GC of previous allocation -- four expensive operations per frame.
- Slide 4: tracker_engine anti-pattern 3 -- `TRT_Preprocessor.process()` does cast, normalize, transpose, and contiguous on CPU then transfers to GPU. All four operations are embarrassingly parallel and perfect for GPU execution.
- Slide 5: tracker_engine anti-pattern 4 -- `.clone().cpu().numpy().tolist()` creates four copies to extract two float values. Use `.item()` for scalars or `.cpu().numpy()` without the clone.
- Slide 6: CRITICAL -- Jetson unified memory -- On Jetson, CPU and GPU share the same physical RAM. There is NO PCIe bus. `cudaMallocManaged` pointers are accessible from both CPU and GPU without explicit transfers. This changes the entire GPU programming model. Many of the anti-patterns above are less costly on Jetson, but fused kernels still help by reducing kernel launch overhead.

### Key Takeaway
- The GPU is fast; PCIe is not. Minimize transfers, not compute. On Jetson, unified memory eliminates the PCIe bottleneck but fused operations still reduce overhead.

## Video 7.2: CUDA Fundamentals (~12 min)

### Slides
- Slide 1: Kernels, threads, and grids -- A CUDA kernel is a `__global__` function that runs on the GPU. Blocks are groups of threads sharing fast shared memory. Grids are collections of all blocks. Grid-stride loops let a single launch handle any array size.
- Slide 2: Thread indexing -- `int idx = blockIdx.x * blockDim.x + threadIdx.x; int stride = blockDim.x * gridDim.x;` Grid-stride loop: `for (int i = idx; i < n; i += stride)`.
- Slide 3: Memory allocation approaches -- Unified memory (`cudaMallocManaged`) uses a single pointer accessible from both CPU and GPU with automatic page migration. Explicit (`cudaMalloc` + `cudaMallocHost`) gives full control and optimal throughput for production pipelines.
- Slide 4: Jetson unified memory deep dive -- On Jetson, unified memory is NOT "simulated" -- CPU and GPU literally share the same physical DRAM. No page migration needed. `cudaMallocManaged` is the natural and often optimal choice on Jetson, unlike desktop GPUs where explicit management usually wins. This simplifies code significantly.
- Slide 5: When NOT to use GPU -- Small data (kernel launch overhead ~5-10 us exceeds compute for <10k elements), irregular access patterns, heavy branching (warp divergence), sequential algorithms, I/O-bound work.

### Live Demo
- Walk through `cuda_basics.cu` showing grid-stride loop pattern. Run the unified vs explicit memory benchmark and discuss results.

### Key Takeaway
- CUDA programming is about managing parallelism (threads/blocks) and memory (unified vs explicit) -- Jetson's unified memory architecture makes this significantly simpler.

## Video 7.3: Fused Kernels and Pinned Memory (~12 min)

### Slides
- Slide 1: The fused preprocessing kernel -- One CUDA kernel replaces three CPU operations (cast uint8 to float32, normalize with mean/std, transpose HWC to CHW). Input: raw uint8 on GPU. Output: normalized CHW float32 on GPU. Zero CPU involvement.
- Slide 2: Performance numbers -- CPU path ~2.1 ms (numpy ops + PCIe transfer). Fused GPU kernel ~0.05 ms (kernel) + ~0.3 ms (initial transfer). With pinned memory + async: transfer overlaps with previous frame's inference.
- Slide 3: Pinned memory -- Regular (pageable) memory can be swapped to disk. Before DMA transfer, CUDA driver copies to pinned staging buffer. Pinned memory eliminates this double-copy: `cudaMallocHost` or `torch.empty(shape, pin_memory=True)`.
- Slide 4: Pre-allocated pinned buffer pool -- Create a pool at startup instead of allocating per-frame. `torch.empty(shape, pin_memory=True)` for each buffer. Acquire/release pattern eliminates per-frame allocation overhead.
- Slide 5: Jetson pinned memory note -- On Jetson, since CPU and GPU share the same physical memory, the distinction between pinned and pageable memory is less impactful for transfer speed. However, pinned memory still prevents the OS from swapping pages, which matters for real-time guarantees. Use `cudaMallocManaged` as the default on Jetson.

### Key Takeaway
- Fused CUDA kernels eliminate CPU-GPU round-trips by combining multiple operations into a single GPU launch -- the biggest win for real-time preprocessing pipelines.

## Video 7.4: CUDA Streams and Batch Inference (~10 min)

### Slides
- Slide 1: CUDA streams -- Default stream executes sequentially. Multiple streams enable overlap: while stream 1 runs inference on frame N, stream 2 transfers frame N+1. `torch.cuda.Stream()` and `with torch.cuda.stream(s):`.
- Slide 2: Stream overlap diagram -- Stream 1: [Transfer N] -> [Preprocess N] -> [Inference N]. Stream 2: [Transfer N+1] -> [Preprocess N+1] -> [Inference N+1]. Overlapped execution doubles throughput.
- Slide 3: Batch inference -- tracker_engine validates candidates one at a time (10 candidates = 10 kernel launches). Batching: `torch.stack(candidates)` then one inference call. Fixed overhead paid once instead of N times.
- Slide 4: Python-level GPU optimization -- `torch.compile` (fuses operations, 1.5-3x speedup), `torch.inference_mode` (stricter than no_grad, disables version counting), AMP autocast (float16 is 2x faster on tensor cores).

### Key Takeaway
- CUDA streams and batching amortize fixed overhead -- streams overlap transfers with compute, batching reduces kernel launch count.

## Video 7.5: CUDA IPC -- Sharing GPU Memory Across Processes (~10 min)

### Slides
- Slide 1: The problem with CPU-mediated GPU IPC -- Process A (GPU) to D2H copy to POSIX shm to H2D copy to Process B (GPU). For 4 MB tensor: ~1 ms wasted on two PCIe round-trips.
- Slide 2: CUDA IPC solution -- Producer calls `cudaIpcGetMemHandle()` to get a 64-byte handle. Consumer calls `cudaIpcOpenMemHandle()` to map the same GPU memory. Zero copies, ~10 us setup.
- Slide 3: Performance comparison -- 4 MB: CUDA IPC 0.07 ms vs pinned D2H+H2D 1.86 ms (15x faster). 64 MB: CUDA IPC ~0.07 ms vs 33.56 ms. IPC cost stays constant regardless of data size.
- Slide 4: Synchronization via IPC events -- `cudaEventCreate` with `cudaEventInterprocess` flag. Producer records event after writing. Consumer waits on event before reading. Prevents data races.
- Slide 5: Jetson CUDA IPC note -- On Jetson with unified memory, CUDA IPC is still useful for multi-process GPU pipelines but the performance advantage over CPU-mediated transfers is smaller since there is no PCIe bus. The synchronization mechanism (IPC events) remains essential for correctness.

### Live Demo
- Run `cuda_ipc_producer` in one terminal and `cuda_ipc_consumer` in another. Show the benchmark comparing IPC vs copy-through-CPU paths.

### Key Takeaway
- CUDA IPC shares GPU memory across processes without any CPU round-trip -- essential for multi-process GPU pipelines where latency matters.
