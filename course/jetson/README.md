# Jetson Developer Guide

This guide covers how NVIDIA Jetson changes the optimization strategies taught
throughout the course. If you are coming from desktop GPU development, many of
the rules you learned are inverted on Jetson. If you are a Python CV/AI
developer deploying to edge hardware for the first time, this guide explains
what is different and why.

## Why Jetson Changes the Optimization Story

On a desktop workstation, the GPU sits across a PCIe bus:

```
CPU (DDR5) ---[ PCIe 4.0 x16: ~25 GB/s ]--- GPU (GDDR6X: ~936 GB/s)
```

The entire course builds around this bottleneck. Lesson 7 exists because PCIe
transfers dominate real-time pipeline latency. Pinned memory, CUDA streams,
and CUDA IPC are all strategies to reduce or eliminate PCIe traffic.

Jetson has no PCIe bus between CPU and GPU. They share the same physical
memory:

```
ARM CPU ---+
           |--- Unified LPDDR5 (Orin: ~205 GB/s)
GPU     ---+
```

This means:

- **`cudaMallocManaged` is not a compromise on Jetson.** On desktop, unified
  memory triggers page faults and driver-mediated migrations across PCIe.
  On Jetson, unified memory is zero-copy access to the same physical DRAM.
  It can match or even beat explicit allocation for many workloads.

- **There is no "PCIe tax."** The dominant lesson from L7 -- that every
  `.cpu()` or `.to(device)` call costs you milliseconds -- does not apply
  in the same way. Transfers between CPU and GPU address spaces are
  effectively free because they are the same address space.

- **Memory bandwidth becomes the bottleneck** instead of transfer latency.
  Jetson's LPDDR5 is shared between CPU and GPU, so bandwidth-hungry GPU
  kernels compete with CPU memory access. Optimizing for memory efficiency
  (smaller data types, fewer loads) matters more than on desktop.

- **Power and thermal limits change the game.** A desktop GPU can sustain
  300W indefinitely. A Jetson Orin tops out at 15-60W depending on the
  power mode. Algorithms that are "fast enough" on desktop may need
  rethinking for Jetson's thermal envelope.

## Jetson Hardware Overview

| Feature | Jetson Nano | Xavier NX | Orin NX | Orin AGX |
|---------|-------------|-----------|---------|----------|
| GPU | 128-core Maxwell | 384-core Volta | 1024-core Ampere | 2048-core Ampere |
| CPU | 4x A57 | 6x Carmel | 8x A78AE | 12x A78AE |
| Memory | 4 GB LPDDR4 | 8 GB LPDDR4x | 8-16 GB LPDDR5 | 32-64 GB LPDDR5 |
| Mem BW | 25.6 GB/s | 51.2 GB/s | 102 GB/s | 205 GB/s |
| AI Perf | 472 GFLOPS | 21 TOPS | 100 TOPS | 275 TOPS |
| TDP | 5-10W | 10-20W | 10-25W | 15-60W |
| JetPack | 4.x (CUDA 10) | 5.x / 6.x | 6.x (CUDA 12) | 6.x (CUDA 12) |

### Which Jetson Maps to Which Course Level

- **Jetson Nano**: Suitable for L1-L3. Limited GPU means CPU optimization
  and IPC patterns matter more. The 128 Maxwell cores are too few for
  complex CUDA kernels, but shared memory IPC is relevant for multi-process
  camera pipelines.

- **Xavier NX**: Covers L1-L7. Volta GPU with tensor cores supports real
  TensorRT inference workloads. The 384 CUDA cores are enough to see
  meaningful speedups from fused kernels and GPU preprocessing.

- **Orin NX / Orin AGX**: Full course coverage L1-L9. Ampere architecture
  with 1024-2048 CUDA cores, DLA (Deep Learning Accelerator), and enough
  memory bandwidth for production multi-stream pipelines. Packaging (L9)
  matters here because Orin is a production deployment target.

## Lesson-by-Lesson Differences

### L1: SIMD -- ARM NEON vs x86 SSE/AVX

The course teaches SIMD using x86 intrinsics (SSE, AVX2, AVX-512) and
xsimd for portability. On Jetson, the CPU is ARM, which uses NEON SIMD:

| Feature | x86 AVX2 | x86 AVX-512 | ARM NEON |
|---------|----------|-------------|----------|
| Register width | 256-bit | 512-bit | 128-bit |
| Float32 lanes | 8 | 16 | 4 |
| Availability | Most desktop | Recent Intel/AMD | All Jetson |

Key differences:

- **Narrower registers.** NEON processes 4 floats per instruction vs 8 for
  AVX2. This means CPU-side vectorization gives smaller speedups on Jetson,
  making GPU offload relatively more attractive.

- **xsimd works on ARM.** The xsimd library used in L1 abstracts over NEON
  automatically. Code written with `xsimd::batch<float>` compiles and runs
  on Jetson without changes -- it just uses NEON under the hood.

- **Auto-vectorization differs.** GCC and Clang have different strengths on
  ARM vs x86. On Jetson, `-march=native` picks up NEON and (on Orin)
  SVE extensions. Always check compiler output with `-fopt-info-vec` or
  `-Rpass=loop-vectorize`.

- **No AVX equivalent.** There is no 256-bit or 512-bit SIMD on ARM Cortex
  A78. If your L1 code relies on wide AVX for throughput, that code path
  does not exist on Jetson. The GPU becomes the primary vector engine.

### L2: Cache Hierarchy

Jetson's ARM cores have a different cache structure:

| Level | Desktop (Zen 4) | Jetson Orin (A78AE) |
|-------|-----------------|---------------------|
| L1 Data | 32 KB / core | 64 KB / core |
| L2 | 1 MB / core | 256 KB / core |
| L3 | 32 MB shared | 2 MB shared |

Key implications:

- **Smaller L2 and L3.** Working sets that fit in desktop L2 (1 MB) may
  spill to L3 or main memory on Jetson. This makes cache-aware tiling
  from L2 even more important.

- **Larger L1.** The 64 KB L1 on A78AE is generous. Hot inner loops that
  fit in L1 perform well.

- **Shared memory bandwidth.** Since CPU and GPU share LPDDR5, a CPU-heavy
  workload can starve the GPU of bandwidth and vice versa. On desktop, CPU
  and GPU have independent memory systems. On Jetson, you must think about
  total system bandwidth.

### L3: Shared Memory and IPC

POSIX shared memory (L3) is even more valuable on Jetson for multi-process
pipelines:

- **No GPU memory isolation benefit.** On desktop, CUDA IPC exists because
  GPU and CPU memory are separate. On Jetson, a `cudaMallocManaged` pointer
  is already accessible from any process that maps it (with appropriate
  IPC handle sharing). The overhead of sharing GPU data is near-zero.

- **Multi-process is the deployment pattern.** Production Jetson systems
  typically run camera capture, inference, and postprocessing as separate
  processes for fault isolation. The seqlock, double-buffer, and cyclic
  buffer patterns from L3 are directly applicable.

- **Power-aware IPC.** Busy-wait synchronization (spinlocks) wastes power
  on a battery- or solar-powered Jetson. Prefer futex-based waiting
  (as in safe-shm) over spin loops.

### L7: GPU Programming -- The Big Inversion

This is where Jetson diverges most from the desktop course material.

**Unified memory is fast on Jetson:**

On desktop, `cudaMallocManaged` triggers page faults when the GPU accesses
CPU-allocated pages (and vice versa). The driver migrates pages across PCIe,
adding milliseconds of latency. The course correctly teaches that explicit
`cudaMalloc` + `cudaMemcpyAsync` with pinned memory outperforms unified
memory on desktop.

On Jetson, `cudaMallocManaged` maps to the same physical LPDDR. There is
no page migration, no PCIe transfer, and no fault handling overhead for
most access patterns. The "explicit is always faster" rule from L7 does
not hold. See `unified-memory-demo.cu` in this directory for a benchmark
that demonstrates the difference.

**CUDA IPC still matters on Jetson:**

Even though there is no PCIe bus, CUDA IPC handles are still the correct
mechanism for sharing GPU memory between processes. The API is the same:
`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`. The difference is that the
underlying operation is essentially free -- it maps an already-accessible
physical address into another process's CUDA context.

**Fused kernels still help, but for different reasons:**

On desktop, fused kernels reduce PCIe round-trips. On Jetson, they reduce:
- Kernel launch overhead (still ~5-10 us per launch)
- Memory bandwidth consumption (fewer passes over data = less pressure on
  shared LPDDR5)
- Cache pollution from intermediate results

**Pinned memory is less critical:**

`cudaMallocHost` (pinned memory) on Jetson allocates from the same LPDDR
pool. It still prevents the OS from swapping the pages, which is useful,
but the "2x transfer speed" benefit from L7 does not apply because there
is no DMA engine staging copy. Use pinned memory on Jetson when you need
to prevent swapping, not for transfer speed.

### L9: Packaging for Jetson

Packaging for Jetson differs from desktop in several ways:

- **Base images.** Desktop uses `ubuntu:22.04` or `nvidia/cuda:12.x`. Jetson
  uses NVIDIA's L4T (Linux for Tegra) images:
  `nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.5-py3` for JetPack 6.

- **No cross-compilation shortcut.** Wheels built on x86 do not run on
  ARM Jetson. You must build on the target architecture (or use QEMU
  emulation, which is slow). CI pipelines need ARM runners or cross-build
  toolchains.

- **JetPack SDK version pinning.** Jetson CUDA, cuDNN, and TensorRT
  versions are tied to the JetPack release. You cannot independently
  upgrade CUDA on Jetson the way you can on desktop. Your Dockerfile
  must target a specific JetPack version.

- **Smaller wheels.** Jetson has limited storage (especially Nano with
  16 GB eMMC). Strip debug symbols, avoid bundling large test data, and
  consider splitting optional dependencies.

See `Dockerfile.jetson` in this directory for a Jetson-specific build
environment.

## Performance Comparison

Measured with a fused preprocessing kernel (640x480 RGB, uint8 to float32
CHW, normalized). Times are kernel execution only, excluding any transfer.

| Metric | Desktop RTX 3090 | Jetson Orin AGX (60W) | Jetson Xavier NX (20W) |
|--------|------------------|-----------------------|------------------------|
| Fused preprocess kernel | 0.05 ms | 0.12 ms | 0.35 ms |
| Explicit alloc + transfer | 0.35 ms | 0.14 ms (*) | 0.38 ms (*) |
| Unified memory overhead | +0.8 ms (page faults) | ~0 ms (zero-copy) | ~0 ms (zero-copy) |
| TensorRT YOLOv8 inference | 1.2 ms | 4.5 ms | 12 ms |
| Full pipeline (30 FPS) | 2.1 ms/frame | 5.8 ms/frame | 16 ms/frame |
| Peak GPU memory BW | 936 GB/s | 205 GB/s | 51.2 GB/s |
| System power draw | ~350W (system) | 15-60W | 10-20W |
| Perf/watt (inference) | ~2.4 infer/J | ~11 infer/J | ~5 infer/J |

(*) On Jetson, "explicit alloc + transfer" involves no actual PCIe
transfer. The cost is allocation overhead and `cudaMemcpy` call dispatch.
The transfer itself is a memcpy within the same LPDDR, which is why unified
memory matches or beats it.

The key takeaway: Jetson is 3-10x slower in raw throughput, but 2-5x better
in performance per watt. For battery-powered or thermally constrained
systems, this is the metric that matters.

## Power and Thermal Optimization Strategy

Jetson supports multiple power modes via `nvpmodel`:

```bash
# List available power modes
sudo nvpmodel --query

# Set to maximum performance (Orin AGX: 60W, all cores active)
sudo nvpmodel -m 0

# Set to power-efficient mode (Orin AGX: 15W, fewer cores)
sudo nvpmodel -m 3

# Maximize clocks within current power budget
sudo jetson_clocks
```

### Optimization strategies for power-constrained Jetson:

1. **Use DLA (Deep Learning Accelerator) for inference.** Orin has two DLA
   engines that run INT8 inference at a fraction of the GPU's power draw.
   TensorRT can target DLA automatically:
   ```python
   config.default_device_type = trt.DeviceType.DLA
   config.DLA_core = 0
   ```

2. **Reduce precision aggressively.** INT8 inference uses ~4x less memory
   bandwidth than FP32 and runs on both GPU and DLA. On Jetson, bandwidth
   is scarce, so precision reduction has outsized impact.

3. **Batch frames when possible.** Kernel launch overhead is a larger
   fraction of total time on Jetson's lower-clocked GPU. Batching
   amortizes this overhead.

4. **Avoid CPU-GPU thrashing.** Even though there is no PCIe bus, frequent
   alternation between CPU and GPU access to the same memory region can
   cause cache coherency overhead. Keep data on one side as long as
   possible.

5. **Monitor thermal throttling.** Jetson throttles clocks when the SoC
   temperature exceeds limits. Use `tegrastats` to monitor:
   ```bash
   tegrastats --interval 1000
   ```

6. **Profile with Nsight Systems on-target.** Nsight Systems supports Jetson
   natively. Profile on the actual hardware, not on a desktop with
   simulated constraints:
   ```bash
   nsys profile --trace=cuda,nvtx python3 my_pipeline.py
   ```

## Files in This Directory

| File | Description |
|------|-------------|
| [README.md](README.md) | This guide |
| [Dockerfile.jetson](Dockerfile.jetson) | Jetson build environment (JetPack 6, L4T) |
| [jetson-benchmarks.py](jetson-benchmarks.py) | Hardware detection and memory benchmark |
| [unified-memory-demo.cu](unified-memory-demo.cu) | Unified vs explicit memory comparison |
