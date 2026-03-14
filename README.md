# ai-cpp-course

**Production C++ for CV/AI Python Developers**

[![Version](https://img.shields.io/badge/version-0.0.1-blue.svg)](VERSION)

A hands-on course for computer vision and algorithm experts who use Python daily
but need to produce production-level performance. Instead of rewriting everything
in C++, learn to surgically replace hot paths and integrate compiled code into
your existing Python workflow.

Real-world examples drawn from [tracker_engine](https://github.com/thebandofficial/tracker_engine)
-- a pure-Python UAV object tracker with measurable performance bottlenecks.

## Lesson Overview

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| L1 | SIMD & Environment Setup | `std::execution::unseq`, pybind11, C++ vs Python perf |
| L2 | Image Processing & Cache Awareness | Cache hierarchy, parallel STL, OpenCV C++ |
| L3 | Shared Memory & IPC | nanobind intro, POSIX shm, lock-free patterns |
| L4 | Nanobind Framework | Zero-copy ndarray, C++ BoundingBox, buffer pools |
| L5 | Python Optimization | `__slots__`, numpy views, pre-allocated buffers, thread pools |
| L6 | Hardware-Level Measurement | `perf_counter_ns`, cache benchmarks, GPU timing |
| L7 | NVIDIA GPU Programming | Fused CUDA kernels, pinned memory, CPU-GPU pipeline |
| L8 | Compile-Time Concepts | C++20 concepts, constexpr LUTs, variant state machines |

## Course Progression

```
L1 SIMD ──> L2 Cache ──> L3 Shared Memory ──> L4 Nanobind
                                                    │
L8 Concepts <── L7 GPU <── L6 Measurement <── L5 Python Opt
```

## Quick Start

Everything runs inside Docker -- no environment pollution.

```bash
# Build the development image
docker build -t ai-cpp-course -f Dockerfile .

# Run the container
docker run -it -v $(pwd):/workspace ai-cpp-course

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

## License

See [LICENSE](LICENSE) for details.
