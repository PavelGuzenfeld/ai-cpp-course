# ai-cpp-course

**Production C++ for CV/AI Python Developers**

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](VERSION)

A hands-on course for computer vision and algorithm experts who use Python daily
but need to produce production-level performance. Instead of rewriting everything
in C++, learn to surgically replace hot paths and integrate compiled code into
your existing Python workflow.

Real-world examples drawn from [tracker_engine](https://github.com/thebandofficial/tracker_engine)
-- a pure-Python UAV object tracker with measurable performance bottlenecks.

## Lesson Overview

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| [L1](ai-cpp-l1/) | SIMD & Environment Setup | `std::execution::unseq`, [pybind11](https://github.com/pybind/pybind11), C++ vs Python perf |
| [L2](ai-cpp-l2/) | Image Processing & Cache Awareness | Cache hierarchy, parallel STL, [OpenCV](https://opencv.org/) C++ |
| [L3](ai-cpp-l3/) | Shared Memory & IPC | [nanobind](https://github.com/wjakob/nanobind) intro, POSIX shm, lock-free patterns |
| [L4](ai-cpp-l4/) | Nanobind Framework | Zero-copy ndarray, C++ BoundingBox, buffer pools |
| [L5](ai-cpp-l5/) | Python Optimization | `__slots__`, numpy views, pre-allocated buffers, thread pools |
| [L6](ai-cpp-l6/) | Hardware-Level Measurement | `perf_counter_ns`, cache benchmarks, GPU timing |
| [L7](ai-cpp-l7/) | GPU Programming — Desktop/Server | Fused CUDA kernels, pinned memory, CUDA IPC, PCIe optimization |
| [L7J](ai-cpp-l7j/) | GPU Programming — Jetson/Edge | Unified memory, power modes, DLA offload, edge deployment |
| [L8](ai-cpp-l8/) | Compile-Time Concepts | C++20 concepts, constexpr LUTs, variant state machines |
| [L9](ai-cpp-l9/) | Going to Production | [scikit-build-core](https://github.com/scikit-build/scikit-build-core) packaging, type stubs, Docker distribution |
| [L10](ai-cpp-l10/) | Profiling-Driven Optimization | The full workflow: profile → identify → optimize → measure |
| [L11](ai-cpp-l11/) | Memory Safety Without Sacrifice | `std::span`, `std::optional`, [ASAN](https://clang.llvm.org/docs/AddressSanitizer.html)/[UBSAN](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html), smart pointers |
| [**Capstone**](capstone/) | **Build a Fast Tracker** | **Reimplement tracker_engine bottlenecks, package as pip library** |

## Course Progression

```
L1 SIMD ──> L2 Cache ──> L3 Shared Memory ──> L4 Nanobind
                                                    │
L8 Concepts <── L7 GPU (Desktop) <── L6 Measurement <── L5 Python Opt
    │           L7J GPU (Jetson)
L9 Packaging ──> L10 Profiling Workflow ──> L11 Memory Safety
                                                    │
                                              Capstone Project
```

## Quick Start

Everything runs inside Docker -- no environment pollution.

```bash
# Build the development image
docker build -t ai-cpp-course -f Dockerfile .

# Run the container
docker run -it -v $(pwd):/workspace ai-cpp-course

# For GPU lessons (L7): build and run the GPU image
docker build -t ai-cpp-course:gpu -f Dockerfile.gpu .
docker run -it --gpus all -v $(pwd):/workspace ai-cpp-course:gpu

# For Jetson (L7J): build on Jetson hardware
docker build -t ai-cpp-course:jetson -f course/jetson/Dockerfile.jetson .
docker run -it --runtime nvidia -v $(pwd):/workspace ai-cpp-course:jetson

# Inside the container: build all lessons
cd /workspace
colcon build
source install/setup.bash
```

## Testing

All tests run inside the Docker container:

```bash
# Inside the container:

# Run all tests
./tests/run_all_tests.sh

# Run Python-only tests (no build required)
pytest ai-cpp-l5/ -v

# Run tests for a specific lesson (after building)
pytest ai-cpp-l4/ -v
pytest ai-cpp-l8/ -v
```

## Course Resources

| Document | Purpose |
|----------|---------|
| [SYLLABUS.md](SYLLABUS.md) | Learning paths, time estimates, prerequisites |
| [CHEATSHEET.md](CHEATSHEET.md) | Single-page reference for all key patterns |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common mistakes and their fixes |
| [assessment/](assessment/) | Pre/post course quiz to measure growth |

## License

See [LICENSE](LICENSE) for details.
