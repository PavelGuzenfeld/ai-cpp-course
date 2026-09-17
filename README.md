# ai-cpp-course

**Production C++ for CV/AI Python Developers**

[![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)](VERSION)

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
| [L2](ai-cpp-l2/) | Image Processing & Cache Awareness | Cache hierarchy, parallel STL, [OpenCV](https://opencv.org/) C++, stride vs width, ASan |
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
| [L12](ai-cpp-l12/) | Compiler Flags & clang-tidy | `-O2`/`-O3`/`-march`/`-ffast-math`, codegen diffs, [clang-tidy](https://clang.llvm.org/extra/clang-tidy/) in CMake, `.clang-tidy` as the policy |
| [L13](ai-cpp-l13/) | Ownership of a C Handle | RAII, move-only types, `= delete` copy, `release()`, the borrowed-pointer double-free |
| [L14](ai-cpp-l14/) | Mocking a Vendor C API | Layout-exact mocks, `static_assert(offsetof(...))`, skip-vs-stub, one impl/two headers |
| [L15](ai-cpp-l15/) | Threading and Atomics | `std::atomic`, memory_order, SPSC ring, `nb::gil_scoped_release`, TSan |
| [L16](ai-cpp-l16/) | Zero-Copy IPC Across Processes | `SCM_RIGHTS` fd passing, shared pages, copy-vs-zero-copy benchmark |
| [L17](ai-cpp-l17/) | Falsifier-First | Kill criteria, cheapest disconfirming experiment first, verdict docs |
| [L18](ai-cpp-l18/) | Golden Oracles and Sanitizers | Independent oracles, mask-disagreement rate, `dlopen`/`LD_PRELOAD`, libFuzzer |
| [L19](ai-cpp-l19/) | Linking Changes Semantics | Static vs shared, ODR, `nm`/`ldd`, `RTLD_LOCAL`, `_GLIBCXX_USE_CXX11_ABI` |
| [L20](ai-cpp-l20/) | Numerical Robustness in Stateful Pipelines | NaN poisoning, coast-don't-update, log-sum-exp, `assert` under `NDEBUG`, UBSan |
| [L21](ai-cpp-l21/) | Speed-of-Light Budgeting | Machine model, `method:` lines, the tax table, FFI crossing cost; then compute/memory/tax floors, budgets as a fraction of SOL, the measured/SOL ratio |
| [**Capstone**](capstone/) | **Build a Fast Tracker** | **Reimplement tracker_engine bottlenecks, package as pip library** |

## Course Progression — the core path

The dependency spine, not the whole course. L12–L21 are self-contained and hang
off it wherever the table above says they do; take them in any order once you
have L4 and L6.

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
# Clone. L3's seven components are submodules -- without --recurse-submodules
# those directories come out empty and L3 builds nothing.
git clone --recurse-submodules https://github.com/PavelGuzenfeld/ai-cpp-course
cd ai-cpp-course

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

# Run the compiled-lesson suites (18 of the 22 lessons; not l1, l9, l10, l12)
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
