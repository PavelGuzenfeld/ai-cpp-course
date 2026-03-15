# Section 0: Course Introduction

## Video 0.1: Welcome and Course Overview (~8 min)

### Slides
- Slide 1: Course title -- "Production C++ for CV/AI Python Developers." Who is teaching, what the course covers at a high level.
- Slide 2: The problem statement -- You write Python daily (numpy, OpenCV, PyTorch). Your code works. But it is too slow for production. You need 30+ FPS on real hardware.
- Slide 3: What this course is NOT -- This is not a "learn C++ from scratch" course. It is a surgical approach: identify Python hot paths, replace them with C++, keep everything else in Python.
- Slide 4: The running example -- tracker_engine, a real-time UAV object tracker written in pure Python. It works but leaves 5-10x performance on the table.
- Slide 5: Course structure overview -- 11 lessons + capstone. Progression from SIMD fundamentals through GPU programming to production packaging. Total ~8-10 hours of video.
- Slide 6: What you will build -- By the end, you will have a pip-installable C++ accelerated tracking library (fast_tracker_utils) that replaces four real bottlenecks with C++ implementations achieving >2x speedup each.

### Key Takeaway
- This course teaches you to surgically optimize Python hot paths with C++, not rewrite everything.

## Video 0.2: Who This Course Is For (~7 min)

### Slides
- Slide 1: Audience profile A -- Python CV/AI developers. You use numpy, OpenCV, and PyTorch daily. You have little or no C++ experience. You need to ship production-level performance.
- Slide 2: Audience profile B -- Jetson/embedded developers. You deploy models on NVIDIA Jetson (Orin, Xavier). You need to maximize throughput on constrained hardware. Unified memory changes your GPU programming model.
- Slide 3: Prerequisites -- Python proficiency (comfortable with numpy, classes, decorators). Basic command line and Docker usage. No C++ experience required -- Lesson 1 introduces the basics.
- Slide 4: Learning paths -- Path A (Python-only, 4-6 hours), Path B (C++ integration, 2-3 days), Path C (complete course, 1 week), Path D (problem-specific lookup table).
- Slide 5: Tools you will use -- Docker (all lessons run in containers), CMake/colcon (build system), pybind11 and nanobind (Python-C++ bindings), perf/ASAN/nsys (profiling and debugging).
- Slide 6: Jetson-specific note -- Throughout the course, look for Jetson callouts. Key differences include unified memory (L7), ARM NEON vs x86 AVX (L1), thermal throttling considerations, and power-constrained optimization strategies.

### Key Takeaway
- Whether you are a Python AI developer or a Jetson embedded engineer, this course meets you where you are and builds toward production-ready C++ integration.

## Video 0.3: Environment Setup (~10 min)

### Slides
- Slide 1: Docker-based development -- Everything runs inside Docker. No environment pollution. Reproducible across machines.
- Slide 2: Building the development image -- `docker build -t ai-cpp-course -f Dockerfile .` and `docker run -it -v $(pwd):/workspace ai-cpp-course`.
- Slide 3: GPU image for Lesson 7 -- `docker build -t ai-cpp-course:gpu -f Dockerfile.gpu .` and `docker run -it --gpus all ...`. Requires NVIDIA Container Toolkit.
- Slide 4: Inside the container -- `colcon build`, `source install/setup.bash`, run tests with `pytest`.
- Slide 5: Jetson setup differences -- On Jetson, you may run natively instead of Docker. JetPack SDK provides CUDA, cuDNN, TensorRT out of the box. The Dockerfile.gpu base image differs (l4t-base vs ubuntu).

### Live Demo
- Build the Docker image from scratch, start the container, run `colcon build`, and execute the Lesson 1 benchmark to verify everything works.

### Key Takeaway
- A working Docker environment is the foundation for every lesson -- invest the time to get it right before proceeding.
