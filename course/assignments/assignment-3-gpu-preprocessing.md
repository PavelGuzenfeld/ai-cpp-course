# Assignment 3: GPU-Accelerated Image Preprocessing

## Objective

Write a fused CUDA kernel that replaces a multi-step Python preprocessing
pipeline, and benchmark it on both desktop GPU and Jetson (if available).

## Background

Lesson 7 showed how tracker_engine wastes time bouncing data between CPU and GPU.
The preprocessing pipeline (cast + normalize + transpose) is a prime target for
a fused CUDA kernel. This assignment has you build it end-to-end.

## Requirements

### Part A: Fused CUDA Kernel (40 points)

Write `preprocess_kernel.cu` containing:

1. `fused_preprocess` — takes uint8 HWC input, produces float32 CHW output:
   - Cast uint8 to float32
   - Divide by 255.0
   - Subtract per-channel mean, divide by per-channel std
   - Transpose HWC to CHW
   - All in a single kernel launch

2. Support arbitrary image sizes (not just 640x480)
3. Use grid-stride loop pattern for flexibility
4. Handle both 3-channel (RGB) and 4-channel (RGBA) inputs

### Part B: Python Bindings (20 points)

Create a nanobind or pybind11 module that exposes:
```python
from gpu_preprocess import preprocess_gpu

# input: numpy uint8 HWC array
# output: numpy float32 CHW array (or torch tensor if torch available)
result = preprocess_gpu(image, mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
```

### Part C: Benchmark (25 points)

Compare three approaches for 640x480, 1280x720, and 1920x1080 images:

1. **NumPy CPU**: `img.astype(float32) / 255; (img - mean) / std; img.transpose(2,0,1)`
2. **PyTorch GPU**: `torch.from_numpy(img).cuda().float() / 255; ...`
3. **Your fused kernel**: single launch, no intermediate buffers

Report kernel time, total time (including transfers), and memory usage.

### Part D: Jetson Comparison (15 points, optional)

On Jetson, the optimization landscape changes:
- Unified memory means `cudaMallocManaged` has no PCIe overhead
- Jetson GPU has fewer CUDA cores but shares memory with CPU
- Power mode affects clock speeds

Run your benchmarks in:
- `nvpmodel -m 0` (MAX performance)
- `nvpmodel -m 1` (15W mode)

Document the differences and explain why.

## Deliverables

- `preprocess_kernel.cu` — fused CUDA kernel
- `preprocess_bindings.cpp` — Python bindings
- `CMakeLists.txt`
- `benchmark_preprocess.py`
- `test_preprocess.py` — correctness tests (compare against NumPy reference)

## Grading

| Criteria | Points |
|----------|--------|
| Kernel produces correct output | 20 |
| Handles RGB and RGBA | 10 |
| Handles arbitrary sizes | 10 |
| Python bindings work | 20 |
| Benchmark shows >5x over NumPy | 15 |
| Tests cover edge cases | 10 |
| Jetson comparison (optional) | 15 |
