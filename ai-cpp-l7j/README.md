# Lesson 7J: GPU Programming on Jetson -- Unified Memory Architecture

## Introduction: Why Jetson Optimization Is Different

Lesson 7 teaches a clear rule: **the PCIe bus is the bottleneck**. Every `.cpu()` call, every `torch.from_numpy().to(device)`, every unnecessary transfer costs you milliseconds because data must cross the PCIe bus between separate CPU and GPU memory systems.

On Jetson, that rule does not apply. There is no PCIe bus. CPU and GPU share the same physical LPDDR5 memory. The optimization story inverts:

| Desktop GPU | Jetson |
|-------------|--------|
| Minimize PCIe transfers | No PCIe -- transfers are free |
| Explicit `cudaMalloc` + pinned memory beats unified | Unified memory matches or beats explicit |
| Memory bandwidth is abundant (936 GB/s on RTX 3090) | Memory bandwidth is scarce (205 GB/s on Orin, shared) |
| Power is unlimited (300W+) | Power is constrained (15-60W) |
| Optimize for throughput | Optimize for throughput per watt |

This lesson revisits the same tracker_engine anti-patterns from [L7](../ai-cpp-l7/) and shows which fixes still apply, which do not, and what new optimization strategies matter on Jetson.

## Jetson Unified Memory Architecture

On a desktop workstation, CPU and GPU have separate memory connected by PCIe:

```
CPU (DDR5) ---[ PCIe 4.0 x16: ~25 GB/s ]--- GPU (GDDR6X: ~936 GB/s)
```

On Jetson, CPU and GPU are on the same SoC and share a single memory pool:

```
ARM CPU (A78AE) ---+
                   |--- Unified LPDDR5
GPU (Ampere)    ---+
                   |
DLA             ---+
```

There is no bus to cross. A pointer allocated with `cudaMallocManaged` is directly accessible from both CPU and GPU without page faults or driver-mediated migration. This is not a convenience wrapper -- it is the hardware's native access mode.

### Jetson Hardware Comparison

| Feature | Jetson Nano | Xavier NX | Orin NX | Orin AGX |
|---------|-------------|-----------|---------|----------|
| GPU | 128-core Maxwell | 384-core Volta | 1024-core Ampere | 2048-core Ampere |
| CPU | 4x A57 | 6x Carmel | 8x A78AE | 12x A78AE |
| Memory | 4 GB LPDDR4 | 8 GB LPDDR4x | 8-16 GB LPDDR5 | 32-64 GB LPDDR5 |
| Memory BW | 25.6 GB/s | 51.2 GB/s | 102 GB/s | 205 GB/s |
| AI Perf | 472 GFLOPS | 21 TOPS | 100 TOPS | 275 TOPS |
| TDP | 5-10W | 10-20W | 10-25W | 15-60W |
| DLA | No | 2x NVDLA 1.0 | 2x NVDLA 2.0 | 2x NVDLA 2.0 |
| JetPack | 4.x (CUDA 10) | 5.x / 6.x | 6.x (CUDA 12) | 6.x (CUDA 12) |

The Nano's 128 Maxwell cores are too few for complex custom CUDA kernels. Xavier NX's 384 Volta cores are enough for real preprocessing and inference workloads. Orin's 1024-2048 Ampere cores with DLA engines are production-grade.

## Revisiting L7 Anti-patterns on Jetson

The [L7 lesson](../ai-cpp-l7/) identifies four tracker_engine anti-patterns. Here is how each one behaves differently on Jetson.

### Anti-pattern 1: `.cpu()` Calls in `prepare_boxes()`

**Desktop**: Each `.cpu()` call triggers a GPU-to-CPU transfer across PCIe, costing ~0.3-0.5 ms per tensor. For two tensors per frame at 30 FPS, this wastes ~20-30 ms per second.

**Jetson**: The `.cpu()` call still copies data, but the copy happens within the same LPDDR5 pool -- effectively a `memcpy`, not a DMA transfer. The cost drops from ~0.5 ms to ~0.01 ms per tensor.

**Verdict**: Still an anti-pattern on Jetson, but low-priority. The copies are cheap. Fix it for code cleanliness (NMS works on CUDA tensors directly), not for performance. Spend your optimization time elsewhere.

### Anti-pattern 2: Per-frame Allocation in `os_tracker_forward()`

**Desktop**: `torch.from_numpy(image).to(device)` allocates CPU memory, allocates GPU memory, transfers, and later garbage collects -- four slow operations per frame.

**Jetson**: Memory allocation is still slow everywhere. `cudaMalloc` must synchronize with the GPU, update page tables, and potentially trigger garbage collection. The transfer is cheap, but the allocation is not.

**Verdict**: Still a serious anti-pattern. Pre-allocate buffers at startup. On Jetson, prefer `cudaMallocManaged` for the pre-allocated buffers -- they are accessible from both CPU and GPU without any copy step.

### Anti-pattern 3: CPU Preprocessing in `TRT_Preprocessor.process()`

**Desktop**: Fused CUDA kernels eliminate three CPU operations and the PCIe transfer. The win is dominated by eliminating the transfer (~2 ms saved).

**Jetson**: There is no transfer to eliminate. The win from fused kernels comes from:
- Reducing kernel launch overhead (~5-10 us per launch, significant on Jetson's lower-clocked GPU)
- Better memory access patterns (one pass over the data instead of four)
- Reduced LPDDR5 bandwidth consumption (critical since CPU and GPU share it)

The improvement is real but smaller. On Orin, expect ~0.3 ms saved versus ~2 ms on desktop.

**Verdict**: Still worth doing, especially on bandwidth-constrained platforms like Xavier NX or Nano. The motivation shifts from "eliminate PCIe transfers" to "minimize memory bandwidth pressure and kernel launches."

### Anti-pattern 4: Copy Chain in `phase_cross_correlation`

```python
result = correlation_tensor.clone().cpu().numpy().tolist()
```

**Desktop**: Four operations, three memory copies, one PCIe crossing.

**Jetson**: The `.clone()` and `.cpu()` are both in-LPDDR copies. Cheaper than desktop, but still three unnecessary memory copies for what should be a single `.item()` call.

**Verdict**: Still wasteful. Use `.item()` for scalars, `.cpu().numpy()` without `.clone()` for small results. The fix is identical to desktop.

### Summary: What Changes and What Does Not

| Anti-pattern | Desktop Fix | Jetson Fix | Priority Change |
|-------------|-------------|------------|-----------------|
| `.cpu()` calls | Drop them (save PCIe transfer) | Drop them (code clarity) | Lower priority |
| Per-frame alloc | Pre-alloc pinned + GPU buffers | Pre-alloc with `cudaMallocManaged` | Same priority |
| CPU preprocessing | Fused CUDA kernel (save transfer) | Fused CUDA kernel (save bandwidth) | Slightly lower |
| Copy chains | Use `.item()` | Use `.item()` | Same priority |

## Unified Memory on Jetson

On desktop, the [L7 lesson](../ai-cpp-l7/) teaches that explicit memory management (`cudaMalloc` + `cudaMallocHost` + `cudaMemcpyAsync`) outperforms unified memory (`cudaMallocManaged`). This is correct for desktop -- unified memory triggers page faults that cross PCIe.

On Jetson, unified memory is the preferred approach. It maps to the same physical LPDDR with zero-copy access from both CPU and GPU.

### Code Comparison: Explicit vs Unified on Jetson

**Explicit approach (desktop-style, unnecessary complexity on Jetson):**

```cuda
// Allocate separate host and device buffers
float* h_input;
float* d_input;
float* d_output;
cudaMallocHost(&h_input, size);    // Pinned host memory
cudaMalloc(&d_input, size);        // Device memory
cudaMalloc(&d_output, size);       // Device memory

// Fill h_input on CPU...
fill_input(h_input, n);

// Copy host to device (on Jetson: memcpy within same LPDDR)
cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

// Launch kernel
my_kernel<<<grid, block>>>(d_input, d_output, n);
cudaDeviceSynchronize();

// Copy result back (on Jetson: another memcpy within same LPDDR)
cudaMemcpy(h_input, d_output, size, cudaMemcpyDeviceToHost);

// Three allocations, two unnecessary copies, three frees
cudaFreeHost(h_input);
cudaFree(d_input);
cudaFree(d_output);
```

**Unified memory approach (natural on Jetson):**

```cuda
// One allocation, accessible from both CPU and GPU
float* data;
float* output;
cudaMallocManaged(&data, size);
cudaMallocManaged(&output, size);

// Fill on CPU -- no copy needed
fill_input(data, n);

// Launch kernel -- GPU reads same physical memory
my_kernel<<<grid, block>>>(data, output, n);
cudaDeviceSynchronize();

// Read result on CPU -- no copy needed
use_result(output, n);

// Two allocations, zero copies, two frees
cudaMallocFree(data);
cudaMallocFree(output);
```

On Jetson, both approaches produce nearly identical performance. The unified version is simpler, has fewer failure modes, and avoids two unnecessary `cudaMemcpy` calls that are just `memcpy` under the hood.

### Performance: Unified vs Explicit on Jetson

Measured with `unified-memory-demo.cu` (see [course/jetson/](../course/jetson/)):

| Metric | Explicit (Jetson Orin) | Unified (Jetson Orin) | Explicit (Desktop RTX 3090) | Unified (Desktop RTX 3090) |
|--------|------------------------|-----------------------|-----------------------------|-----------------------------|
| Allocation | 0.08 ms | 0.06 ms | 0.05 ms | 0.04 ms |
| Transfer | 0.14 ms (*) | 0 ms | 0.35 ms | +0.8 ms (faults) |
| Kernel | 0.12 ms | 0.12 ms | 0.05 ms | 0.05 ms |
| **Total** | **0.34 ms** | **0.18 ms** | **0.45 ms** | **0.89 ms** |

(*) On Jetson, "transfer" is a `memcpy` within LPDDR5, not a DMA across PCIe.

On desktop, explicit wins by 2x. On Jetson, unified wins because there is no transfer to optimize away -- you are just skipping unnecessary copies.

## Power and Thermal Management

A desktop GPU can sustain 300W+ indefinitely. Jetson Orin tops out at 60W in its highest power mode and can be configured as low as 15W. This changes benchmarking methodology and optimization priorities.

### nvpmodel: Power Mode Control

Jetson supports multiple power modes that trade performance for power consumption:

```bash
# List available power modes
sudo nvpmodel --query

# Set to maximum performance (Orin AGX: 60W, all 12 CPU cores, full GPU clocks)
sudo nvpmodel -m 0

# Set to power-efficient mode (Orin AGX: 15W, fewer cores, lower clocks)
sudo nvpmodel -m 3

# Lock clocks at maximum within current power budget
sudo jetson_clocks

# Restore dynamic clock scaling
sudo jetson_clocks --restore
```

On Orin AGX, the power modes are roughly:

| Mode | TDP | CPU Cores | GPU Clocks | Use Case |
|------|-----|-----------|------------|----------|
| MAXN (0) | 60W | 12 | Max | Benchmarking, AC-powered |
| 50W (1) | 50W | 12 | Reduced | Continuous operation |
| 30W (2) | 30W | 8 | Reduced | Battery, light thermal load |
| 15W (3) | 15W | 4 | Minimum | Low-power standby |

### tegrastats: Real-time Monitoring

```bash
# Monitor every second
tegrastats --interval 1000
```

Output includes:
- CPU and GPU utilization per core
- Memory usage (shared pool)
- GPU and CPU temperatures
- Current power draw (in milliwatts)
- Clock frequencies

Example output:
```
RAM 4563/62845MB GR3D_FREQ 76% CPU [43%@2201,51%@2201,38%@2201,45%@2201]
GPU 1300MHz EMC 3199MHz TEMP CPU@48.5C GPU@47.2C SOC@46.8C POM_5V_GPU 8234mW
```

### Benchmarking at Different Power Levels

Always benchmark at the power level you will deploy at. A kernel that meets your frame budget at 60W may miss it at 15W:

```bash
# Benchmark at deployment power level
sudo nvpmodel -m 2           # Set to 30W mode
sudo jetson_clocks            # Lock clocks for consistent results
sleep 5                       # Let thermal state stabilize
python3 jetson-benchmarks.py  # Run benchmarks

# Compare with max performance
sudo nvpmodel -m 0
sudo jetson_clocks
sleep 5
python3 jetson-benchmarks.py
```

Thermal throttling adds another variable. After sustained load, the SoC temperature rises and clocks are reduced automatically. Always let benchmarks run for at least 60 seconds to observe steady-state performance, not just burst performance.

## Jetson-Specific Optimizations

These optimizations are either unique to Jetson or have significantly different tradeoffs on Jetson compared to desktop.

### DLA (Deep Learning Accelerator) Offload

Orin and Xavier have dedicated DLA engines that run INT8 and FP16 inference at a fraction of the GPU's power draw. DLA frees the GPU for other work (custom CUDA kernels, preprocessing) while inference runs in parallel on dedicated hardware.

```python
import tensorrt as trt

builder = trt.Builder(logger)
config = builder.create_builder_config()

# Target DLA instead of GPU
config.default_device_type = trt.DeviceType.DLA
config.DLA_core = 0  # Use DLA core 0 (Orin has 2)

# DLA requires INT8 or FP16
config.set_flag(trt.BuilderFlag.INT8)
config.set_flag(trt.BuilderFlag.FP16)

# Fallback to GPU for layers DLA does not support
config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
```

DLA limitations:
- Only supports a subset of TensorRT layers (convolution, pooling, elementwise, etc.)
- Requires INT8 or FP16 precision -- no FP32
- Lower peak throughput than the GPU, but significantly better perf/watt
- Two DLA cores on Orin can run two models or two instances simultaneously

### INT8/FP16 Precision for Tensor Cores

Reduced precision has outsized impact on Jetson because memory bandwidth is the scarce resource:

| Precision | Bytes/element | BW usage (relative) | Tensor core support |
|-----------|---------------|---------------------|---------------------|
| FP32 | 4 | 1.0x | No |
| FP16 | 2 | 0.5x | Yes (2x throughput) |
| INT8 | 1 | 0.25x | Yes (4x throughput) |

On desktop with 936 GB/s bandwidth, the difference between FP32 and INT8 is nice to have. On Jetson Orin with 205 GB/s shared between CPU and GPU, it can mean the difference between hitting and missing your frame budget.

TensorRT handles precision reduction during engine build:

```python
config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16 (safe for most models)
config.set_flag(trt.BuilderFlag.INT8)  # Enable INT8 (requires calibration)
```

### CUDA Graphs for Reducing Kernel Launch Overhead

On desktop, kernel launch overhead (~5-10 us) is negligible relative to PCIe transfer time. On Jetson, where transfers are free and GPU clocks are lower, launch overhead becomes a larger fraction of total frame time.

CUDA Graphs capture a sequence of kernel launches and replay them with a single launch command, eliminating per-kernel CPU-side overhead:

```cuda
// Capture phase: record the sequence of operations
cudaGraph_t graph;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// Your normal kernel launches
preprocess_kernel<<<grid1, block1, 0, stream>>>(input, temp, n);
normalize_kernel<<<grid2, block2, 0, stream>>>(temp, output, n);
postprocess_kernel<<<grid3, block3, 0, stream>>>(output, result, n);

cudaStreamEndCapture(stream, &graph);

// Instantiate the graph (one-time cost)
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

// Replay phase: launch all three kernels with one API call
// Saves ~10-20 us per frame (2-3 kernel launches worth of overhead)
cudaGraphLaunch(graphExec, stream);
```

For a pipeline with 5-10 kernel launches per frame at 30 FPS, CUDA Graphs can save 1.5-3 ms per second -- meaningful on Jetson.

### Multi-process Pipelines with CUDA IPC

CUDA IPC (covered in [L7](../ai-cpp-l7/) and [L3](../ai-cpp-l3/) for the IPC fundamentals) is still the correct mechanism for sharing GPU memory between processes on Jetson. The API is identical:

```cuda
// Producer process
cudaIpcMemHandle_t handle;
cudaIpcGetMemHandle(&handle, d_data);
// Share handle via POSIX shared memory (see L3)

// Consumer process
float* d_shared;
cudaIpcOpenMemHandle((void**)&d_shared, handle, cudaIpcMemLazyEnablePeerAccess);
// d_shared points to producer's memory -- zero copy on Jetson
```

The difference from desktop: on desktop, CUDA IPC avoids two PCIe crossings (saving ~1 ms for 4 MB). On Jetson, CUDA IPC avoids two in-LPDDR copies (saving ~0.01 ms). The performance benefit is minimal, but CUDA IPC remains important for **process isolation** -- separating camera capture, inference, and postprocessing into independent processes so that a crash in one does not take down the others.

For the IPC handle sharing mechanism, refer to the POSIX shared memory patterns in [L3](../ai-cpp-l3/).

## Benchmarking on Jetson

Use `jetson-benchmarks.py` (see [course/jetson/](../course/jetson/)) to measure hardware-specific characteristics. The script auto-detects the Jetson model and runs memory bandwidth, kernel launch, and inference benchmarks.

### Expected Performance: Jetson vs Desktop

| Benchmark | Desktop RTX 3090 | Orin AGX (60W) | Xavier NX (20W) | Nano (10W) |
|-----------|------------------|----------------|-----------------|------------|
| Fused preprocess (640x480) | 0.05 ms | 0.12 ms | 0.35 ms | 1.2 ms |
| TensorRT YOLOv8 (FP16) | 1.2 ms | 4.5 ms | 12 ms | N/A |
| TensorRT YOLOv8 (INT8) | 0.8 ms | 2.8 ms | 8 ms | N/A |
| DLA YOLOv8 (INT8) | N/A | 6.2 ms | 15 ms | N/A |
| Full pipeline (30 FPS) | 2.1 ms/frame | 5.8 ms/frame | 16 ms/frame | 33 ms/frame |
| System power | ~350W | 15-60W | 10-20W | 5-10W |
| Perf/watt (inferences/J) | ~2.4 | ~11 | ~5 | ~2 |

Key observations:
- Orin is 3-4x slower than RTX 3090 in raw throughput but 4-5x better in perf/watt
- DLA inference is slower than GPU but frees the GPU for other work and uses less power
- INT8 provides ~1.6x speedup over FP16 on both desktop and Jetson, but the bandwidth savings matter more on Jetson
- Nano cannot run complex models like YOLOv8 at real-time rates; it is suitable for simpler models

## Build and Run

Jetson builds use a different base image and must run on ARM. There is no cross-compilation shortcut -- CUDA wheels built on x86 do not run on ARM Jetson.

### Building on Jetson Natively

```bash
# Using the Jetson Dockerfile
docker build -f course/jetson/Dockerfile.jetson -t ai-cpp-jetson .
docker run --runtime nvidia -it ai-cpp-jetson

# Inside the container
cd /workspace
colcon build --packages-select ai_cpp_l7j
source install/setup.bash
```

### Running Benchmarks

```bash
# Detect hardware and run memory benchmarks
python3 course/jetson/jetson-benchmarks.py

# Run the unified memory demo (compare unified vs explicit on Jetson)
./build/unified-memory-demo

# Run the fused preprocessing kernel benchmark
python3 ai-cpp-l7/benchmark_gpu.py  # Same benchmark, different results on Jetson
```

### Verifying Power Mode

Always verify your power mode before benchmarking:

```bash
sudo nvpmodel --query          # Show current mode
tegrastats --interval 1000     # Monitor in real time
```

## Exercises

1. Run `unified-memory-demo` from [course/jetson/](../course/jetson/) on Jetson hardware (or Orin emulation). Compare the unified vs explicit results against the desktop numbers from [L7](../ai-cpp-l7/)'s `cuda_basics.cu`. Explain why the relative performance is inverted.

2. Take the fused preprocessing kernel from [L7](../ai-cpp-l7/) (`gpu_preprocess.cu`) and benchmark it on Jetson at three different power modes (MAXN, 30W, 15W). Plot throughput vs power draw. At which power mode does the kernel first fail to meet 30 FPS?

3. Modify the [L7](../ai-cpp-l7/) CUDA IPC demo to run on Jetson. Measure the IPC handle open time and compare against the desktop numbers. Why is the gap between IPC and copy-through-CPU much smaller on Jetson?

4. Wrap the fused preprocessing kernel in a CUDA Graph. Measure the per-frame overhead reduction compared to launching the kernel directly. Run at 30 FPS for 60 seconds and compare average frame times.

5. (Advanced) Build a TensorRT engine for a simple model (e.g., ResNet-18) targeting both GPU and DLA. Compare inference latency and power consumption. Use `tegrastats` to measure power draw during sustained inference on each accelerator.

6. (Advanced) Implement a multi-process pipeline on Jetson: Process A captures frames and does preprocessing (fused kernel), Process B runs inference (TensorRT on DLA), Process C does postprocessing (CPU). Use CUDA IPC (from [L7](../ai-cpp-l7/)) and POSIX shared memory (from [L3](../ai-cpp-l3/)) for inter-process communication. Monitor total system power with `tegrastats`.

## What You Learned

- Jetson's unified memory architecture eliminates the PCIe bottleneck that dominates desktop GPU optimization
- `cudaMallocManaged` is the preferred allocation strategy on Jetson -- it provides zero-copy access from both CPU and GPU
- The L7 anti-patterns remain anti-patterns on Jetson, but the priority shifts: per-frame allocation and kernel launch overhead matter more than transfer elimination
- Power and thermal constraints replace PCIe bandwidth as the primary optimization target
- DLA offload, INT8 precision, and CUDA Graphs are high-impact optimizations specific to Jetson deployment
- Always benchmark at your deployment power mode -- burst performance at 60W does not predict steady-state behavior at 15W
- CUDA IPC remains useful on Jetson for process isolation, even though the performance benefit over copy-through-CPU is minimal

## Lesson Files

| File | Description |
|------|-------------|
| [README.md](README.md) | This lesson (Jetson GPU programming) |
| [../ai-cpp-l7/cuda_basics.cu](../ai-cpp-l7/cuda_basics.cu) | CUDA fundamentals: grid-stride, unified vs explicit memory |
| [../ai-cpp-l7/gpu_preprocess.cu](../ai-cpp-l7/gpu_preprocess.cu) | Fused CUDA preprocessing kernel |
| [../ai-cpp-l7/gpu_preprocess_cpu.cpp](../ai-cpp-l7/gpu_preprocess_cpu.cpp) | CPU reference preprocessing implementation |
| [../ai-cpp-l7/cuda_ipc_producer.cu](../ai-cpp-l7/cuda_ipc_producer.cu) | CUDA IPC producer (shared with L7) |
| [../ai-cpp-l7/cuda_ipc_consumer.cu](../ai-cpp-l7/cuda_ipc_consumer.cu) | CUDA IPC consumer (shared with L7) |
| [../ai-cpp-l7/benchmark_gpu.py](../ai-cpp-l7/benchmark_gpu.py) | CPU vs GPU performance comparison |
| [../ai-cpp-l7/benchmark_cuda_ipc.py](../ai-cpp-l7/benchmark_cuda_ipc.py) | IPC vs CPU-mediated transfer benchmark |
| [../course/jetson/unified-memory-demo.cu](../course/jetson/unified-memory-demo.cu) | Unified vs explicit memory on Jetson |
| [../course/jetson/jetson-benchmarks.py](../course/jetson/jetson-benchmarks.py) | Jetson hardware detection and benchmarks |
| [../course/jetson/Dockerfile.jetson](../course/jetson/Dockerfile.jetson) | Jetson build environment (JetPack 6, L4T) |
