# Section 7J: GPU Programming — Jetson/Edge (Unified Memory Architecture)

> This section is for students deploying on NVIDIA Jetson. It covers the same CUDA
> fundamentals as Section 7 but with Jetson-specific memory strategies, power management,
> and edge deployment techniques.

## Video 7J.1: Why Jetson Is Different (~10 min)

### Slides
- Slide 1: Desktop vs Jetson architecture -- Desktop: CPU (DDR5) --[PCIe 25 GB/s]-- GPU (GDDR6X 936 GB/s). Jetson: CPU + GPU share unified LPDDR5 (~205 GB/s on Orin).
- Slide 2: No PCIe bus means no PCIe tax. `cudaMallocManaged` is zero-copy on Jetson. The entire L7 desktop lesson is built around avoiding PCIe transfers -- on Jetson, that problem does not exist.
- Slide 3: What IS the bottleneck on Jetson? Memory bandwidth (shared between CPU and GPU), compute capacity (fewer CUDA cores), and power/thermal limits (15-60W vs desktop 300W+).
- Slide 4: Jetson hardware lineup -- Orin AGX (2048 cores, 64GB, 60W), Orin NX (1024 cores, 16GB, 25W), Xavier NX (384 cores, 8GB, 15W), Nano (128 cores, 4GB, 10W). The course content scales across all of these.
- Slide 5: tracker_engine on Jetson -- The same UAV tracking system, but on a drone-mounted Jetson Orin. Different constraints, different optimization priorities.

### Key Takeaway
- Jetson inverts the desktop GPU optimization story: unified memory is fast, but you are now constrained by power, thermal limits, and shared memory bandwidth.

## Video 7J.2: Unified Memory on Jetson (~12 min)

### Slides
- Slide 1: `cudaMallocManaged` on desktop vs Jetson -- Desktop: page faults cross PCIe, driver migrates pages. Jetson: same physical RAM, zero-copy, no migration needed.
- Slide 2: Code comparison -- Explicit path (cudaMalloc + cudaMemcpy) vs unified path (cudaMallocManaged). On Jetson, unified is simpler AND competitive in performance.
- Slide 3: When explicit still wins on Jetson -- Pinning prevents OS from swapping pages (matters for real-time). Explicit gives predictable allocation timing. Large allocations may benefit from explicit to avoid OS page management.
- Slide 4: Best practice for Jetson -- Default to `cudaMallocManaged`. Use explicit allocation only when you need hard real-time guarantees or profiling shows a bottleneck.

### Live Demo
- Run `unified-memory-demo.cu` on Jetson. Show that unified and explicit are close in performance (unlike desktop where explicit wins by 2-5x).

### Key Takeaway
- On Jetson, `cudaMallocManaged` is the natural choice. It simplifies code with minimal performance cost.

## Video 7J.3: Fused Kernels on Jetson (~10 min)

### Slides
- Slide 1: Fused kernels still help on Jetson -- not because of PCIe savings, but because they reduce kernel launch overhead and improve memory access patterns on the shared memory bus.
- Slide 2: The same `fused_preprocess_kernel` from L7 works on Jetson unchanged. But the speedup vs CPU is smaller because there is no PCIe transfer to eliminate.
- Slide 3: Memory bandwidth optimization matters more -- Jetson's LPDDR5 is shared. GPU kernels that are bandwidth-hungry compete with CPU. Use smaller data types (INT8, FP16) to reduce bandwidth pressure.
- Slide 4: Tensor cores and DLA -- Jetson Orin has tensor cores for FP16/INT8. The DLA (Deep Learning Accelerator) can run inference independently, freeing the GPU for other work.

### Key Takeaway
- Fused kernels reduce launch overhead and bandwidth on Jetson. Combine with FP16/INT8 to maximize throughput within the power budget.

## Video 7J.4: Power and Thermal Management (~10 min)

### Slides
- Slide 1: Power modes -- `nvpmodel` controls CPU/GPU clock speeds and core counts. MAXN = maximum performance, 15W/25W = power-limited modes. Performance scales roughly linearly with power.
- Slide 2: `tegrastats` -- real-time monitoring of GPU/CPU utilization, memory, power draw, temperature. Essential for understanding if you are compute-bound, memory-bound, or thermal-throttled.
- Slide 3: Thermal throttling -- Jetson will reduce clocks when temperature exceeds limits. Sustained workloads at MAXN may throttle. Active cooling or power-limited modes give more consistent performance.
- Slide 4: Optimization strategy for edge -- Profile at your target power mode, not MAXN. A pipeline that runs at 30 FPS at MAXN but throttles to 20 FPS after 2 minutes is not production-ready.
- Slide 5: CUDA graphs -- Pre-record a sequence of kernel launches, replay with minimal host overhead. Especially valuable on Jetson where CPU cores are weaker and kernel launch overhead is proportionally higher.

### Live Demo
- Run `power_mode_demo.py` -- show current power mode, run benchmark, show tegrastats output.

### Key Takeaway
- Always profile at your production power mode. Use `tegrastats` to distinguish between compute-bound, memory-bound, and thermal-throttled workloads.

## Video 7J.5: Multi-Process Pipelines and Deployment (~10 min)

### Slides
- Slide 1: Multi-process GPU pipelines on Jetson -- camera capture, preprocessing, inference, postprocessing as separate processes. Process isolation gives fault tolerance.
- Slide 2: CUDA IPC on Jetson -- Still useful for process isolation and synchronization, but the performance gain over CPU-mediated transfer is smaller (no PCIe to save). The IPC event synchronization mechanism remains essential.
- Slide 3: DeepStream -- NVIDIA's framework for video analytics pipelines on Jetson. GStreamer-based, zero-copy between plugins. Consider for production video pipelines.
- Slide 4: Deployment checklist -- Dockerfile.jetson, JetPack version pinning, power mode configuration, thermal validation, tegrastats monitoring in production.

### Key Takeaway
- Jetson deployment requires power-mode testing, thermal validation, and production monitoring that desktop GPUs do not need. Plan for these from the start.
