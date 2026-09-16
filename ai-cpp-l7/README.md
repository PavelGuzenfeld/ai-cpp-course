# Lesson 7: GPU Programming — Desktop/Server (PCIe Architecture)

> **Jetson developers**: This lesson covers desktop/server GPUs with discrete PCIe-attached memory.
> If you are deploying on NVIDIA Jetson, see [Lesson 7J](../ai-cpp-l7j/) for the unified memory
> architecture where many of these strategies change fundamentally.

## The Real Problem: PCIe Is the Bottleneck

Most CV/Python developers think "I have a GPU, so my code is fast." Wrong. Having a GPU means nothing if you keep bouncing data between CPU and GPU.

The numbers tell the story:
- **PCIe 3.0 x16 bandwidth**: ~12 GB/s (theoretical peak)
- **GPU memory bandwidth (e.g., RTX 3090)**: ~936 GB/s
- **Ratio**: GPU memory is **~75x faster** than the PCIe bus

Every time you call `.cpu()`, `.numpy()`, or `torch.from_numpy().to(device)`, you're paying the PCIe tax. For a real-time tracker running at 30+ FPS, this kills your frame budget.

## tracker_engine: A Case Study in What Not to Do

tracker_engine is a real-time UAV tracking system built on [TensorRT](https://developer.nvidia.com/tensorrt). It works. It's also leaving 5-10x performance on the table because of constant CPU↔GPU data bouncing.

### Anti-pattern 1: `prepare_boxes()` — CPU-side NMS

```python
def prepare_boxes(predictions, device):
    boxes = predictions[:, :4].cpu()  # <-- GPU→CPU transfer
    scores = predictions[:, 4].cpu()  # <-- GPU→CPU transfer
    keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.5)  # NMS on CPU!
    return predictions[keep]
```

**Problem**: NMS is highly parallelizable. `torchvision.ops.nms` supports GPU tensors directly — the `.cpu()` calls are unnecessary. Every frame, two tensors cross the PCIe bus for no reason.

**Fix**: Drop the `.cpu()` calls. `torchvision.ops.nms` works on CUDA tensors.

### Anti-pattern 2: `os_tracker_forward()` — Per-frame allocation

```python
def os_tracker_forward(image, device):
    tensor = torch.from_numpy(image).to(device)  # allocate + transfer every frame
    # ... inference ...
```

**Problem**: `torch.from_numpy()` creates a new CPU tensor, then `.to(device)` allocates GPU memory and copies. Every. Single. Frame. This means:
1. CPU-side memory allocation (slow)
2. GPU-side memory allocation (slow)
3. PCIe transfer of the data (slow)
4. Previous GPU allocation gets garbage collected (slow)

**Fix**: Pre-allocate a pinned memory buffer and a GPU tensor once. Copy into the pinned buffer, then use `tensor.copy_()` for async transfer.

### Anti-pattern 3: `TRT_Preprocessor.process()` — Three-step CPU preprocessing

```python
def process(self, image):
    image = image.astype(np.float32)       # Step 1: cast (CPU)
    image = (image - self.mean) / self.std  # Step 2: normalize (CPU)
    image = image.transpose(2, 0, 1)       # Step 3: HWC→CHW (CPU)
    image = np.ascontiguousarray(image)     # Step 4: ensure contiguous (CPU)
    # then copy to GPU for TensorRT inference
```

**Problem**: Four CPU operations, then a transfer. The GPU is sitting idle waiting for preprocessed data. All of these operations — cast, normalize, transpose — are embarrassingly parallel and perfect for GPU execution.

**Fix**: A single fused CUDA kernel that takes uint8 HWC and produces float32 CHW normalized output directly on GPU. See `gpu_preprocess.cu`.

### Anti-pattern 4: `phase_cross_correlation` — The Copy Chain

```python
result = correlation_tensor.clone().cpu().numpy().tolist()
```

**Problem**: Four operations, three memory copies:
1. `.clone()` — GPU→GPU copy (unnecessary if you're about to leave the GPU)
2. `.cpu()` — GPU→CPU transfer
3. `.numpy()` — Creates numpy view (cheap, but only works on CPU)
4. `.tolist()` — Converts to Python list (slow, allocates Python objects)

**Fix**: If you need a single scalar, use `.item()`. If you need a small result, use `.cpu().numpy()` (skip the clone). If you need the data for further GPU work, keep it on GPU.

## CUDA Fundamentals

Before diving into the tracker fixes, here are the GPU programming building blocks.

### Kernels, Threads, and Grids

A CUDA **kernel** is a function that runs on the GPU. The `__global__` specifier marks it as callable from CPU code:

```cuda
__global__ void vector_add(const float* a, const float* b, float* c, int n)
{
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride)  // grid-stride loop
        c[i] = a[i] + b[i];
}
```

Key terms:
- **Block**: a group of threads that share fast shared memory (`blockDim.x` threads)
- **Grid**: the collection of all blocks launched for a kernel (`gridDim.x` blocks)
- **Grid-stride loop**: each thread processes multiple elements, so a single launch handles any array size

```
Grid:  [Block 0: 256 threads] [Block 1: 256 threads] ... [Block N]
        Thread 0..255           Thread 256..511
```

### Memory Allocation: Unified vs Explicit

| Approach | Syntax | Performance | Use case |
|----------|--------|-------------|----------|
| **Unified Memory** | `cudaMallocManaged(&ptr, size)` | Automatic migration, simpler code | Prototyping, irregular access |
| **Explicit + Pinned** | `cudaMalloc` + `cudaMallocHost` | Full control, optimal throughput | Production pipelines |

On desktop/server GPUs, unified memory triggers page faults across PCIe, so a carefully tuned explicit pipeline with `cudaMemcpyAsync` will outperform it. On Jetson, the opposite is true — see [L7J](../ai-cpp-l7j/).

See [`cuda_basics.cu`](cuda_basics.cu) for a complete comparison of both approaches with benchmarks.

## Solution 1: Fused CUDA Kernels

Instead of three separate CPU operations for preprocessing, write one CUDA kernel:

```cuda
__global__ void fused_preprocess_kernel(
    const uint8_t* __restrict__ input,   // HWC uint8
    float* __restrict__ output,          // CHW float32
    int height, int width, int channels,
    const float* __restrict__ mean,
    const float* __restrict__ std)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = height * width * channels;
    if (idx >= total) return;

    int c = idx % channels;
    int w = (idx / channels) % width;
    int h = idx / (channels * width);

    float pixel = static_cast<float>(input[h * width * channels + w * channels + c]);
    pixel = (pixel / 255.0f - mean[c]) / std[c];

    // Write in CHW order
    output[c * height * width + h * width + w] = pixel;
}
```

This single kernel replaces `TRT_Preprocessor.process()` entirely. The image goes from raw uint8 on GPU to normalized CHW float32 on GPU with zero CPU involvement.

**Performance**: For a 640x480x3 image:
- CPU path: ~2.1ms (numpy ops + PCIe transfer)
- Fused GPU kernel: ~0.05ms (kernel) + ~0.3ms (initial transfer if needed)
- With pinned memory + async: the transfer overlaps with previous frame's inference

## Solution 2: Pinned Memory

Regular (pageable) memory can be swapped to disk by the OS. Before a DMA transfer to GPU, the CUDA driver must first copy pageable memory to a pinned (page-locked) staging buffer. This doubles the transfer time.

```
Pageable:  CPU alloc → copy to pinned staging → DMA to GPU  (2 copies)
Pinned:    CPU alloc (pinned) → DMA to GPU                   (1 copy)
```

### Using pinned memory:

```cpp
// C++ with CUDA
void* ptr;
cudaMallocHost(&ptr, size);  // Pinned allocation
// ... use ptr ...
cudaFreeHost(ptr);            // Must free with cudaFreeHost
```

```python
# PyTorch
pinned_tensor = torch.empty(shape, pin_memory=True)
# Transfer is now ~2x faster
gpu_tensor.copy_(pinned_tensor, non_blocking=True)  # Async!
```

### Pre-allocated pool (what tracker_engine should do):

Instead of allocating per-frame, create a pool of pinned buffers at startup:

```python
class PinnedBufferPool:
    def __init__(self, n_buffers, shape):
        self.buffers = [torch.empty(shape, pin_memory=True) for _ in range(n_buffers)]
        self.available = list(range(n_buffers))

    def acquire(self):
        idx = self.available.pop()
        return self.buffers[idx], idx

    def release(self, idx):
        self.available.append(idx)
```

This eliminates per-frame allocation overhead in `os_tracker_forward()`.

## Solution 3: CUDA Streams

By default, all CUDA operations go into the default stream and execute sequentially. With multiple streams, you can overlap:

```
Default stream (sequential):
  [Transfer frame N] → [Preprocess N] → [Inference N] → [Transfer frame N+1] → ...

Multiple streams (overlapped):
  Stream 1: [Transfer N  ] → [Preprocess N  ] → [Inference N  ]
  Stream 2:     [Transfer N+1] → [Preprocess N+1] → [Inference N+1]
```

```python
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

with torch.cuda.stream(stream1):
    gpu_tensor.copy_(pinned_input, non_blocking=True)
    output = model(gpu_tensor)

# While stream1 runs inference, stream2 can prepare next frame
with torch.cuda.stream(stream2):
    next_gpu_tensor.copy_(next_pinned_input, non_blocking=True)
```

## Your kernel is not the only thing on the device

Everything above assumes your component is alone on the GPU. On an edge box it
is not: a detector, a tracker and a preprocessor share one device, and each was
probably measured in isolation by whoever wrote it. A component can be correct
on its own and harmful in composition.

The question is which of a neighbour's choices actually reaches you.
[`stream_contention_probe.cu`](stream_contention_probe.cu) runs two components
at once — B is latency-sensitive and short, A is a long-running throughput
stage — and varies only what A does. Measured on an Orin NX (8 SMs), B's
per-call latency over 2000 calls, four runs:

| A's behaviour | B p50 (ms) | B p99 (ms) | B p99 vs alone |
|---|---|---|---|
| no neighbour at all | 0.0149 | 0.0177 | — |
| own stream, `cudaStreamSynchronize` | 0.0148 | 0.0175–0.0214 | ×0.99–1.21 |
| own stream, `cudaDeviceSynchronize` | 0.0149 | 0.0176–0.0197 | ×0.99–1.12 |
| **the legacy default stream** | 1.5580 | **1.5667–1.5690** | **×88** |

The result is not the one the folklore predicts. `cudaDeviceSynchronize()` in
a neighbour is nearly free here — it blocks *A's host thread*, and blocking a
thread you do not own costs you nothing. What costs 88× is A launching into
the legacy default stream, because that stream implicitly synchronises with
every other blocking stream in the process. A did not call a "device-wide"
anything; it just failed to create a stream.

So the rule is not "avoid `cudaDeviceSynchronize`". It is:

**Every component owns a stream. A library that launches into the default
stream is a latency bug for everyone else in the process, and it will not
show up in that library's own benchmark.**

That last clause is the reason this is hard to catch. A's numbers are
unchanged in all four rows — A is fine. The damage is entirely in someone
else's p99, and neither team is measuring the pair.

Note the p50/p99 split: in the default-stream row B's *median* is already
1.56 ms, so here the tail and the middle move together. Report both anyway —
contention that only shows in the tail is the common case, and a mean would
have hidden the 0.0177 → 0.0214 row entirely.

### The same bug wearing a library's clothes

A neighbour does not have to be your code. NVIDIA's NPP has a legacy
stream-setting call that is **process-global**: one component setting it
clobbers the binding another component is relying on, with no diagnostic. The
fix is the context-object API, which scopes the stream to a call instead of to
the process.

That is second-hand evidence, not something this lesson reproduces — it was
root-caused in `gst-nvmm-cpp`
([`95dbf60`](https://github.com/PavelGuzenfeld/gst-nvmm-cpp/commit/95dbf60),
[`6b2ae01`](https://github.com/PavelGuzenfeld/gst-nvmm-cpp/commit/6b2ae01)),
where a device-wide synchronisation in one component also stalled a
concurrently running inference stream
([`10fb7e3`](https://github.com/PavelGuzenfeld/gst-nvmm-cpp/commit/10fb7e3)).
Worth reading as a class of bug rather than an NPP fact: **any library call
that sets process-global state is a composition hazard**, and the API that
looks convenient is usually the one that is global.

Before adopting a library into a pipeline that already has a GPU stage, the
question to ask is not "is it fast" but "what does it set that I do not own".

## Solution 4: Batch Inference

tracker_engine's `track_restoration` validates candidate detections one at a time:

```python
# tracker_engine: one-by-one (bad)
for candidate in candidates:
    score = model.validate(candidate)  # One GPU inference per candidate
```

Each inference call has fixed overhead: kernel launch, memory allocation, synchronization. With 10 candidates, you pay this overhead 10 times.

```python
# Batched (good)
batch = torch.stack(candidates)  # Combine into one tensor
scores = model.validate(batch)   # One GPU inference for all
```

Batching amortizes the fixed overhead and allows the GPU to utilize more of its parallel hardware.

## Python-Level GPU Optimization

### torch.compile (PyTorch 2.0+)

```python
model = torch.compile(model)  # JIT compiles the model graph
```

This fuses operations, eliminates Python overhead, and can provide 1.5-3x speedup on post-processing code that uses multiple PyTorch ops.

### torch.inference_mode

```python
with torch.inference_mode():
    output = model(input)
```

Stricter than `torch.no_grad()` — disables autograd entirely, saving memory and compute. Always use for inference workloads.

### Automatic Mixed Precision (AMP)

```python
with torch.cuda.amp.autocast():
    output = model(input)  # Uses float16 where safe
```

Half-precision is 2x faster on tensor cores and uses half the memory bandwidth.

## Solution 5: CUDA IPC — Sharing GPU Memory Across Processes

In [L3](../ai-cpp-l3/) you learned CPU-side IPC with POSIX shared memory. But what if both processes use the GPU? The naive approach copies data through the CPU:

```
Process A (GPU) → D2H copy → POSIX shm → H2D copy → Process B (GPU)
         ~0.5ms        ~0µs        ~0.5ms
```

For a 4MB tensor, that's ~1ms wasted on two PCIe round-trips. **CUDA IPC eliminates both copies** by letting Process B map Process A's GPU memory directly:

```
Process A (GPU) → IPC handle → Process B (GPU)
         0 copies, ~10µs setup
```

### How It Works

1. **Producer** allocates GPU memory and gets an IPC handle:
   ```cuda
   float* d_data;
   cudaMalloc(&d_data, size);
   // ... fill d_data with a kernel ...

   cudaIpcMemHandle_t handle;
   cudaIpcGetMemHandle(&handle, d_data);
   // Share 'handle' (64 bytes) via POSIX shm, pipe, socket, etc.
   ```

2. **Consumer** opens the handle and gets a pointer to the *same* GPU memory:
   ```cuda
   float* d_shared;
   cudaIpcOpenMemHandle((void**)&d_shared, handle,
                         cudaIpcMemLazyEnablePeerAccess);
   // d_shared points to producer's GPU memory — zero copy!
   ```

3. **Synchronization** via IPC events prevents data races:
   ```cuda
   // Producer: signal data is ready
   cudaEvent_t event;
   cudaEventCreate(&event, cudaEventInterprocess | cudaEventDisableTimingPeer);
   cudaEventRecord(event);

   cudaIpcEventHandle_t evt_handle;
   cudaIpcGetEventHandle(&evt_handle, event);
   // Share evt_handle with consumer

   // Consumer: wait for producer's event
   cudaEvent_t remote_event;
   cudaIpcOpenEventHandle(&remote_event, evt_handle);
   cudaStreamWaitEvent(myStream, remote_event);
   ```

### Performance: IPC vs CPU Round-Trip

Measured on NVIDIA GeForce RTX 3060 Laptop GPU:

| Data size | CUDA IPC | Pinned D2H+H2D | Pageable D2H+H2D |
|-----------|----------|-----------------|-------------------|
| 4 MB  | 0.07 ms | 1.86 ms | 1.01 ms |
| 16 MB | ~0.07 ms | 7.13 ms | 4.32 ms |
| 64 MB | ~0.07 ms | 33.56 ms | 31.07 ms |

CUDA IPC is effectively free — once the handle is opened, the pointer works like any device pointer. For 4MB the IPC demo measured a **15x speedup** over the copy-through-CPU path. For larger data the gap widens further since IPC cost stays constant.

### When to Use CUDA IPC

- **Multi-process GPU pipelines**: camera capture → preprocessing → inference → postprocessing, each in its own process for fault isolation
- **Producer-consumer with GPU data**: one process generates GPU tensors, another consumes them
- **Shared inference results**: multiple consumers need the same detection output without duplicating GPU memory

### Limitations

- Both processes must be on the **same GPU** (or use NVLink peer access)
- The producer must keep its `cudaMalloc` alive while consumers use the handle
- Not supported on all platforms (requires Linux with compatible driver)
- The IPC handle is 64 bytes — it must be transmitted via some side channel (POSIX shm, socket, etc.)

See [`cuda_ipc_producer.cu`](cuda_ipc_producer.cu) and [`cuda_ipc_consumer.cu`](cuda_ipc_consumer.cu) for a complete working example, and [`benchmark_cuda_ipc.py`](benchmark_cuda_ipc.py) for the PyTorch-based transfer comparison.

## When NOT to Use GPU

GPUs are not universally faster. Avoid GPU for:

- **Small data**: Kernel launch overhead (~5-10μs) exceeds compute time for small arrays. If your tensor has fewer than ~10,000 elements, CPU is likely faster.
- **Irregular access patterns**: GPUs need coalesced memory access. Scatter/gather operations with random indices waste bandwidth.
- **Heavy branching**: GPU threads in a warp must execute the same instruction. If your code has many if/else branches, threads diverge and serialize.
- **Sequential algorithms**: Some algorithms are inherently sequential (e.g., certain graph traversals). These won't benefit from GPU parallelism.
- **I/O-bound work**: If you're waiting on disk or network, GPU won't help.

## Key Takeaways

1. **The GPU is fast; PCIe is not.** Minimize transfers, not compute.
2. **Fuse operations** to avoid round-trips. One kernel doing three things beats three kernels.
3. **Pre-allocate everything.** Per-frame allocation is a performance antipattern.
4. **Use pinned memory** for any host→device transfer path.
5. **Batch your inference.** Fixed overhead × N is worse than fixed overhead × 1.
6. **Profile before optimizing.** Use `torch.cuda.Event` for timing (see [L6](../ai-cpp-l6/)), [`nsys`](https://developer.nvidia.com/nsight-systems) for system-level analysis.

## Build and Run

```bash
# Inside Docker container
cd /workspace
colcon build --packages-select ai_cpp_l7
source install/setup.bash

# Run benchmarks (GPU optional — falls back to CPU)
python3 ai-cpp-l7/benchmark_gpu.py

# Run the pipeline comparison demo
python3 ai-cpp-l7/gpu_pipeline_demo.py
```

## Exercises

1. Build and run `gpu_preprocess.cu` — compare against `gpu_preprocess_cpu.cpp` using `benchmark_gpu.py`
2. Modify the pinned allocator pool size and measure the impact on sustained throughput
3. Run `gpu_pipeline_demo.py` to see the "wrong way" vs "right way" pipeline comparison
4. (Advanced) Add CUDA stream overlap to the GPU pipeline demo
5. Profile with [`nsys`](https://developer.nvidia.com/nsight-systems): Run `nsys profile python3 benchmark_gpu.py` and examine the CPU/GPU timeline. Where are the gaps?
6. Build and run `cuda_basics.cu` — compare unified memory vs explicit pinned memory performance
7. Run the CUDA IPC demo: start `cuda_ipc_producer` in one terminal, `cuda_ipc_consumer` in another. Compare the IPC path vs copy-through-CPU numbers
8. Run `benchmark_cuda_ipc.py` to see the transfer overhead comparison across data sizes
9. Build and run `stream_contention_probe.cu` on your own device. Do you
   reproduce the 88× default-stream penalty, and is `cudaDeviceSynchronize`
   as cheap for you as it is on an Orin NX? Then give A enough blocks to fill
   every SM and re-run: all four rows collapse to the same number. Explain
   why that version of the experiment cannot answer the question, and write
   the verdict using [L17](../ai-cpp-l17/)'s template.

## What You Learned

- PCIe is the bottleneck, not GPU compute — minimize transfers
- Fused CUDA kernels combine multiple operations into one GPU launch
- Pinned memory provides ~2x faster host-to-device transfers
- CUDA streams enable overlapping transfers and compute
- Batch inference amortizes fixed overhead across multiple inputs
- Not all workloads benefit from GPU — small data, branchy code, I/O-bound work stays on CPU
- CUDA IPC shares GPU memory across processes without CPU round-trips — essential for multi-process GPU pipelines
- A component measured alone can be harmful in composition, and the damage
  lands in someone else's p99 — so the pair has to be measured, not each half
- Every component owns a stream: launching into the legacy default stream cost
  a neighbour 88× on an Orin NX, while a neighbour's `cudaDeviceSynchronize`
  cost it essentially nothing
- A library call that sets process-global state is a composition hazard; ask
  what a dependency sets that you do not own

## Lesson Files

| File | Description |
|------|-------------|
| [cuda_basics.cu](cuda_basics.cu) | CUDA fundamentals: grid-stride, unified vs explicit memory |
| [cuda_ipc_producer.cu](cuda_ipc_producer.cu) | CUDA IPC producer: exports GPU memory handle |
| [cuda_ipc_consumer.cu](cuda_ipc_consumer.cu) | CUDA IPC consumer: maps producer's GPU memory |
| [gpu_preprocess.cu](gpu_preprocess.cu) | Fused CUDA preprocessing kernel |
| [gpu_preprocess_cpu.cpp](gpu_preprocess_cpu.cpp) | CPU reference preprocessing implementation |
| [pinned_allocator.cpp](pinned_allocator.cpp) | Pinned memory pool with fallback |
| [batch_inference_demo.py](batch_inference_demo.py) | Batched GPU inference demonstration |
| [cuda_streams_demo.py](cuda_streams_demo.py) | CUDA streams overlap demonstration |
| [gpu_pipeline_demo.py](gpu_pipeline_demo.py) | Wrong vs right GPU pipeline comparison |
| [tracker_engine_fixes.py](tracker_engine_fixes.py) | Tracker engine GPU anti-pattern fixes |
| [benchmark_gpu.py](benchmark_gpu.py) | CPU vs GPU performance comparison |
| [benchmark_cuda_ipc.py](benchmark_cuda_ipc.py) | IPC vs CPU-mediated transfer benchmark |
| [stream_contention_probe.cu](stream_contention_probe.cu) | Two components sharing a device; what a neighbour's stream choice costs (host `nvcc`, not in the CMake build) |
| [CMakeLists.txt](CMakeLists.txt) | CMake build configuration with CUDA support |
| [test_gpu.py](test_gpu.py) | Unit tests for preprocess and allocator |
| [test_integration_gpu.py](test_integration_gpu.py) | Full pipeline and batch inference tests |
