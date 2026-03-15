# Production C++ for CV/AI Python Developers

## Subtitle

Stop rewriting everything in C++. Learn to surgically replace Python hot paths with high-performance C++ and GPU code.

## Course Description

You write Python for computer vision and AI. Your algorithms work. But they are too slow for production: 30 FPS tracking on a drone, real-time inference on a Jetson edge device, or low-latency video analytics on a GPU server.

The conventional advice is "rewrite it in C++." That is the wrong answer. The right answer is to keep your Python workflow and replace only the bottlenecks with compiled code that Python calls seamlessly.

This course teaches you exactly how to do that.

### What You Will Learn

- **Identify real bottlenecks** using profiling tools, not guesswork
- **Write C++ extensions** that Python imports like any other module (pybind11, nanobind)
- **Use CUDA kernels** to fuse multi-step preprocessing into a single GPU launch
- **Share data between processes** using zero-copy shared memory (POSIX shm, CUDA IPC)
- **Package and distribute** your C++ extensions as pip-installable wheels
- **Deploy on NVIDIA Jetson** with unified memory optimizations specific to edge devices

### Real-World Focus

Every example comes from tracker_engine — a real UAV object tracking system with measurable performance problems. You will fix real anti-patterns, not toy examples:

- A preprocessing pipeline that wastes 2ms per frame bouncing data between CPU and GPU
- Per-frame memory allocation that triggers garbage collection during inference
- String-based state machines that burn CPU on every frame
- Copy-heavy history buffers that defeat the cache hierarchy

By the end, you will have built fast_tracker_utils — a pip-installable package that makes the tracker 3-10x faster, ready for production deployment on desktop GPUs or NVIDIA Jetson.

### Who This Course Is For

- Python developers working in computer vision, robotics, or ML inference
- Engineers deploying AI models on NVIDIA Jetson (Nano, Xavier, Orin)
- Anyone who needs "C++ speed" without abandoning their Python workflow
- Prerequisites: comfortable with Python and NumPy; no C++ experience required

### What Is Included

- 11 progressive lessons + capstone project
- Docker environment — zero setup, works on any machine
- Jetson-specific Dockerfile and benchmarks
- 4 graded coding assignments
- 300+ automated tests to verify your work
- Real benchmark numbers on desktop GPU and Jetson hardware

### Technical Requirements

- A computer with Docker installed (Linux, macOS, or Windows with WSL2)
- NVIDIA GPU recommended but not required (CPU fallback for all exercises)
- For Jetson lessons: NVIDIA Jetson device with JetPack 6.x (optional)
