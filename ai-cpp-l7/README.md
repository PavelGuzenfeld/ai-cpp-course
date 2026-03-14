# Lesson 7: NVIDIA GPU Programming — Keeping Data Where It Belongs

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

## What You Learned

- PCIe is the bottleneck, not GPU compute — minimize transfers
- Fused CUDA kernels combine multiple operations into one GPU launch
- Pinned memory provides ~2x faster host-to-device transfers
- CUDA streams enable overlapping transfers and compute
- Batch inference amortizes fixed overhead across multiple inputs
- Not all workloads benefit from GPU — small data, branchy code, I/O-bound work stays on CPU

## Lesson Files

| File | Description |
|------|-------------|
| [gpu_preprocess.cu](gpu_preprocess.cu) | Fused CUDA preprocessing kernel |
| [gpu_preprocess_cpu.cpp](gpu_preprocess_cpu.cpp) | CPU reference preprocessing implementation |
| [pinned_allocator.cpp](pinned_allocator.cpp) | Pinned memory pool with fallback |
| [batch_inference_demo.py](batch_inference_demo.py) | Batched GPU inference demonstration |
| [cuda_streams_demo.py](cuda_streams_demo.py) | CUDA streams overlap demonstration |
| [gpu_pipeline_demo.py](gpu_pipeline_demo.py) | Wrong vs right GPU pipeline comparison |
| [tracker_engine_fixes.py](tracker_engine_fixes.py) | Tracker engine GPU anti-pattern fixes |
| [benchmark_gpu.py](benchmark_gpu.py) | CPU vs GPU performance comparison |
| [CMakeLists.txt](CMakeLists.txt) | CMake build configuration with CUDA support |
| [test_gpu.py](test_gpu.py) | Unit tests for preprocess and allocator |
| [test_integration_gpu.py](test_integration_gpu.py) | Full pipeline and batch inference tests |
