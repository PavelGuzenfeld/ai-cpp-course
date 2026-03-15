# Section 7 Quiz: NVIDIA GPU Programming

## Q1: GPU memory bandwidth is approximately 75x faster than PCIe bandwidth. What is the main implication for real-time pipelines?

- a) You should always use the GPU for every operation
- b) Minimizing CPU-GPU data transfers matters more than optimizing GPU compute kernels
- c) You should use PCIe 5.0 to eliminate the bottleneck
- d) GPU memory is 75x larger than CPU memory

**Answer: b)** Since the PCIe bus is the bottleneck (not GPU compute), the most impactful optimization is reducing the number and size of transfers between CPU and GPU. Keeping data on the GPU as long as possible avoids paying the PCIe tax.

## Q2: Why is calling `.cpu()` on GPU tensors before `torchvision.ops.nms` an anti-pattern?

- a) NMS cannot run on CPU
- b) `torchvision.ops.nms` supports CUDA tensors directly, so the `.cpu()` calls force two unnecessary PCIe transfers per frame for no reason
- c) `.cpu()` corrupts the tensor data
- d) NMS is faster on CPU than GPU

**Answer: b)** `torchvision.ops.nms` works on both CPU and GPU tensors. Moving tensors to CPU with `.cpu()` wastes time on GPU-to-CPU transfers and forces the result back to GPU afterward -- two PCIe round-trips that serve no purpose.

## Q3: What problem does pinned (page-locked) memory solve for GPU transfers?

- a) It compresses data before transfer
- b) It eliminates the extra copy from pageable memory to a staging buffer that the CUDA driver must perform before DMA, roughly doubling transfer speed
- c) It allows the GPU to access CPU memory directly without any transfer
- d) It encrypts data during transfer for security

**Answer: b)** Regular pageable memory can be swapped to disk by the OS, so the CUDA driver must first copy it to a pinned staging buffer before DMA. Pinned memory skips this intermediate copy, reducing the transfer to a single DMA operation.

## Q4: What does a fused CUDA preprocessing kernel accomplish compared to doing preprocessing on the CPU?

- a) It reduces the accuracy of preprocessing for speed
- b) It combines multiple operations (uint8-to-float conversion, normalization, HWC-to-CHW transpose) into a single GPU kernel, avoiding CPU computation and PCIe transfers
- c) It uses the CPU and GPU simultaneously for each operation
- d) It automatically selects the best interpolation method

**Answer: b)** Instead of performing cast, normalize, and transpose as separate CPU numpy operations and then transferring the result to GPU, a fused kernel does all three in one GPU pass. The image goes from raw uint8 to normalized CHW float32 entirely on the GPU.

## Q5: On a Jetson device with shared CPU/GPU memory, why is minimizing transfers still important even though CPU and GPU access the same physical memory?

- a) It is not important -- shared memory eliminates all overhead
- b) Even with unified memory, cache coherency protocols, memory access serialization, and the overhead of launching separate small kernels still degrade performance
- c) Jetson devices do not support CUDA
- d) Shared memory is slower than PCIe on Jetson

**Answer: b)** While Jetson's unified memory eliminates PCIe transfers, the CPU and GPU still contend for memory bandwidth, and cache coherency protocols add overhead. Fusing operations and minimizing kernel launches remains critical for meeting real-time deadlines on power-constrained hardware.

## Q6: Why does batching multiple inference inputs into a single call improve GPU performance?

- a) It reduces the model's parameter count
- b) It amortizes fixed per-launch overhead (kernel dispatch, memory setup, synchronization) across all inputs and allows better GPU utilization
- c) It automatically reduces precision from float32 to float16
- d) It forces the GPU to use more cores

**Answer: b)** Each GPU inference call has fixed overhead: kernel launch, memory allocation, and synchronization. Running 10 individual inferences pays this overhead 10 times. Batching pays it once and lets the GPU's parallel hardware process all inputs simultaneously.

## Q7: What does CUDA IPC (Inter-Process Communication) enable that POSIX shared memory cannot?

- a) Sharing CPU memory between processes
- b) Direct sharing of GPU memory between processes on the same GPU, avoiding two PCIe round-trips (device-to-host and host-to-device)
- c) Communication between processes on different machines
- d) Automatic load balancing between processes

**Answer: b)** CUDA IPC lets one process export a GPU memory handle that another process can open to access the same GPU memory directly. Without CUDA IPC, sharing GPU data between processes requires copying to CPU via PCIe, passing through POSIX shared memory, and copying back to GPU -- two expensive transfers eliminated by CUDA IPC.

## Q8: Which of the following workloads would NOT benefit from GPU acceleration?

- a) Resizing a batch of 100 images simultaneously
- b) Sorting a list of 50 integers
- c) Running a neural network on a 640x480 image
- d) Computing element-wise operations on a 10-million-element tensor

**Answer: b)** GPU kernel launch overhead is 5-10 microseconds. For 50 integers, the CPU can complete the sort in less time than it takes to launch a GPU kernel. Small data sizes cannot generate enough parallel work to offset the launch overhead.
