# Section 7J Quiz: GPU Programming — Jetson/Edge

## Q1: Why is `cudaMallocManaged` performant on Jetson but not on desktop GPUs?

- a) Jetson has faster DRAM
- b) Jetson CPU and GPU share the same physical memory — no PCIe page faults
- c) Jetson driver optimizes unified memory differently
- d) Desktop GPUs do not support unified memory

**Answer: b)** On Jetson, CPU and GPU share the same physical LPDDR5. There is no PCIe bus, so unified memory pointers access the same physical RAM without page faults or migrations.

## Q2: What is the primary bottleneck on Jetson that replaces the PCIe bottleneck on desktop?

- a) CPU clock speed
- b) GPU compute throughput
- c) Shared memory bandwidth between CPU and GPU
- d) Network I/O

**Answer: c)** On Jetson, CPU and GPU share the same memory bus. Bandwidth-hungry GPU kernels compete with CPU memory access, making shared bandwidth the new bottleneck.

## Q3: You are running a tracker at 30 FPS on Jetson Orin at MAXN mode. After 3 minutes, FPS drops to 22. What is the most likely cause?

- a) Memory leak
- b) Thermal throttling — the GPU reduced clocks due to temperature
- c) Python garbage collection
- d) CUDA driver bug

**Answer: b)** Sustained high-power workloads on Jetson cause thermal throttling. The GPU reduces clock speeds to stay within thermal limits, reducing throughput.

## Q4: Which tool shows real-time GPU utilization, power draw, and temperature on Jetson?

- a) nvidia-smi
- b) htop
- c) tegrastats
- d) nvprof

**Answer: c)** `tegrastats` is the Jetson-specific monitoring tool that shows GPU/CPU utilization, memory usage, power draw, and temperature in real time.

## Q5: Do fused CUDA kernels (like the preprocessing kernel from L7) still help on Jetson?

- a) No — there is no PCIe transfer to eliminate
- b) Yes — they reduce kernel launch overhead and improve memory access patterns
- c) Only for images larger than 4K
- d) Only when using explicit memory allocation

**Answer: b)** While fused kernels do not save PCIe transfer time on Jetson, they still reduce kernel launch overhead (significant on Jetson's weaker CPU) and improve memory bandwidth utilization through better access patterns.

## Q6: What does `nvpmodel -m 0` do on a Jetson Orin?

- a) Puts the device in sleep mode
- b) Sets maximum performance mode (all cores, max clocks)
- c) Enables power saving mode (15W)
- d) Resets the GPU

**Answer: b)** Mode 0 (MAXN) enables all CPU/GPU cores at maximum clock frequencies for peak performance, at the cost of higher power consumption.

## Q7: Why should you profile your Jetson application at the production power mode rather than MAXN?

- a) MAXN is not accurate
- b) Applications that meet latency targets at MAXN may fail at production power modes due to lower clocks
- c) MAXN uses more memory
- d) Production power mode has different CUDA APIs

**Answer: b)** If your production deployment uses 15W or 25W mode, profiling at MAXN gives misleadingly good numbers. A pipeline that runs at 30 FPS at MAXN may only achieve 20 FPS at 15W.

## Q8: What is the DLA on Jetson Orin and why is it useful?

- a) A display adapter for monitor output
- b) A Deep Learning Accelerator that runs inference independently of the GPU, freeing GPU for other tasks
- c) A memory controller for faster DRAM access
- d) A debug logging agent

**Answer: b)** The DLA (Deep Learning Accelerator) on Jetson Orin can run neural network inference in parallel with the GPU. This lets you offload inference to the DLA while using the GPU for preprocessing or other compute tasks.
