# Course Syllabus

## Who This Course Is For

Computer vision and algorithm engineers who:
- Write Python daily (numpy, OpenCV, PyTorch)
- Have little or no C++ experience
- Need to ship production-level performance
- Want to optimize without rewriting everything

## Learning Paths

### Path A: Python-Only (4-6 hours)

No C++ build required. Focus on optimizing Python itself.

```
L5 Python Optimization (1.5h)
  └─> L10 Profiling Workflow (1.5h)
        └─> L6 Measurement (partial — Python timing only) (1h)
              └─> Capstone baseline benchmark (0.5h)
```

**You'll learn**: `__slots__`, numpy views, pre-allocated buffers, thread pools,
the complete profile-optimize-measure workflow, Amdahl's law.

### Path B: C++ Integration (2-3 days)

Full course for integrating C++ into Python workflows.

```
Day 1: Foundations
  L1 SIMD (1h) → L2 Cache (1.5h) → L5 Python Optimization (1.5h)

Day 2: C++ Bindings + Measurement
  L4 Nanobind (1.5h) → L6 Measurement (1.5h) → L3 Shared Memory (1h)

Day 3: Advanced + Production
  L7 GPU (1.5h) → L8 Compile-Time Concepts (1.5h) → L9 Packaging (1h)
```

### Path C: Complete Course (1 week)

All lessons including memory safety and capstone.

```
Day 1: L1, L2, L5
Day 2: L4, L6, L3
Day 3: L7, L8
Day 4: L9, L10, L11
Day 5: Capstone project
```

### Path D: "I Have a Specific Problem"

| Problem | Go to |
|---------|-------|
| "My Python loop is slow" | [L1](ai-cpp-l1/) (SIMD), [L5](ai-cpp-l5/) (Python opt) |
| "Too many allocations per frame" | [L5](ai-cpp-l5/) (pre-alloc), [L4](ai-cpp-l4/) (C++ buffer pool) |
| "GPU inference is slow" | [L7](ai-cpp-l7/) (fused kernels, pinned memory) |
| "How do I measure what's slow?" | [L6](ai-cpp-l6/) (measurement), [L10](ai-cpp-l10/) (workflow) |
| "How do I ship this to production?" | [L9](ai-cpp-l9/) (packaging) |
| "I keep getting segfaults in C++" | [L11](ai-cpp-l11/) (memory safety) |
| "My state machine is string-based" | [L8](ai-cpp-l8/) (variant + visit) |
| "I need inter-process communication" | [L3](ai-cpp-l3/) (shared memory) |

## Lesson Details

| # | Lesson | Time | Prerequisites | Build Required |
|---|--------|------|---------------|----------------|
| [L1](ai-cpp-l1/) | SIMD & Environment | 1h | None | Docker + colcon |
| [L2](ai-cpp-l2/) | Image Processing & Cache | 1.5h | L1 | Docker + colcon |
| [L3](ai-cpp-l3/) | Shared Memory & IPC | 1h | L1, L4 | Docker + colcon |
| [L4](ai-cpp-l4/) | Nanobind Framework | 1.5h | L1 | Docker + colcon |
| [L5](ai-cpp-l5/) | Python Optimization | 1.5h | None | **None** (pure Python) |
| [L6](ai-cpp-l6/) | Hardware Measurement | 1.5h | L4 (for C++ timer) | Docker + colcon |
| [L7](ai-cpp-l7/) | NVIDIA GPU Programming | 1.5h | L4, L6 | Docker + colcon + CUDA (optional) |
| [L8](ai-cpp-l8/) | Compile-Time Concepts | 1.5h | L4 | Docker + colcon |
| [L9](ai-cpp-l9/) | Production Packaging | 1h | L4 | Docker |
| [L10](ai-cpp-l10/) | Profiling Workflow | 1.5h | L5, L6 | **None** (pure Python) |
| [L11](ai-cpp-l11/) | Memory Safety | 1h | L4 | Docker + colcon |
| [Cap](capstone/) | Capstone Project | 3-4h | All | Docker + colcon |

## Assessment

Use `assessment/pre_assessment.py` before starting and `assessment/post_assessment.py`
after completing the course to measure your growth.

## Getting Started

```bash
docker build -t ai-cpp-course -f Dockerfile .
docker run -it -v $(pwd):/workspace ai-cpp-course
cd /workspace
colcon build
source install/setup.bash
```

Then follow your chosen learning path above.
