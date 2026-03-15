# Section 7: GPU Programming — Desktop/Server (PCIe Architecture)

> Students deploying on Jetson should take Section 7J instead or in addition to this section.

## Video 7.1: PCIe Is the Bottleneck (~10 min)

### Slides
- Slide 1: The real problem -- PCIe 3.0 x16 bandwidth ~12 GB/s vs GPU memory bandwidth ~936 GB/s (RTX 3090). GPU memory is 75x faster than the PCIe bus. Every `.cpu()`, `.numpy()`, or `.to(device)` call pays the PCIe tax.
- Slide 2: Desktop GPU architecture diagram -- CPU (DDR5) connected to GPU (GDDR6X) via PCIe. All data must cross this bus. This is NOT how Jetson works (see Section 7J).
- Slide 3: tracker_engine anti-patterns overview -- Four common mistakes that cause unnecessary PCIe transfers. Each one costs 0.5-2ms per frame.
- Slide 4: Anti-pattern walkthrough -- `.cpu()` before NMS, per-frame `.to(device)`, CPU preprocessing before GPU inference, `.clone().cpu().numpy().tolist()` copy chain.

### Key Takeaway
- On desktop GPUs, the PCIe bus is the bottleneck, not GPU compute. Every CPU-GPU transfer costs real milliseconds.

## Video 7.2: CUDA Fundamentals (~12 min)

### Slides
- Slide 1: Kernels, threads, and grids -- `__global__` functions, blockIdx/threadIdx, grid-stride loops.
- Slide 2: Thread indexing code -- `int idx = blockIdx.x * blockDim.x + threadIdx.x; int stride = blockDim.x * gridDim.x;`
- Slide 3: Memory allocation on desktop -- Unified memory (`cudaMallocManaged`) triggers page faults across PCIe. Explicit (`cudaMalloc` + `cudaMallocHost`) gives full control. On desktop, explicit wins for production pipelines.
- Slide 4: When NOT to use GPU -- Small data (<10k elements), irregular access, heavy branching, sequential algorithms, I/O-bound work.

### Live Demo
- Walk through `cuda_basics.cu`. Run unified vs explicit benchmark -- show that explicit wins on desktop.

### Key Takeaway
- On desktop GPUs, explicit memory management outperforms unified memory due to PCIe page-fault overhead.

## Video 7.3: Fused Kernels and Pinned Memory (~12 min)

### Slides
- Slide 1: Fused preprocessing kernel -- one CUDA kernel replaces three CPU operations (cast, normalize, transpose). Zero CPU involvement.
- Slide 2: Performance -- CPU path ~2.1ms vs fused kernel ~0.05ms + ~0.3ms transfer. 6x faster.
- Slide 3: Pinned memory -- pageable requires double-copy via staging buffer. Pinned memory = single DMA transfer = 2x faster.
- Slide 4: Pre-allocated pinned buffer pool -- eliminate per-frame allocation. Acquire/release pattern.

### Key Takeaway
- Fused kernels + pinned memory = the two biggest wins for desktop GPU preprocessing pipelines.

## Video 7.4: CUDA Streams and Batch Inference (~10 min)

### Slides
- Slide 1: CUDA streams overlap transfers with compute. While stream 1 runs inference, stream 2 transfers next frame.
- Slide 2: Batch inference -- amortize kernel launch overhead. 10 candidates = 1 launch instead of 10.
- Slide 3: Python-level optimization -- `torch.compile`, `torch.inference_mode`, AMP autocast.

### Key Takeaway
- Streams and batching amortize fixed overhead across multiple operations.

## Video 7.5: CUDA IPC — Sharing GPU Memory Across Processes (~10 min)

### Slides
- Slide 1: Problem -- cross-process GPU sharing via CPU costs 2 PCIe round-trips (~1ms for 4MB).
- Slide 2: CUDA IPC -- `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`. Zero copies, ~10us setup.
- Slide 3: Benchmarks -- 4MB: IPC 0.07ms vs D2H+H2D 1.86ms (15x). 64MB: 0.07ms vs 33.56ms.
- Slide 4: IPC event synchronization for correctness.

### Live Demo
- Run `cuda_ipc_producer` + `cuda_ipc_consumer` side by side.

### Key Takeaway
- CUDA IPC eliminates PCIe round-trips for multi-process GPU pipelines -- the gains are massive on desktop where PCIe is the bottleneck.
