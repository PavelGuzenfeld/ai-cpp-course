# Section 12: Capstone Project -- Build a Fast Tracker

## Video 12.1: Project Overview and Baseline Profiling (~10 min)

### Slides
- Slide 1: What you will build -- `fast_tracker_utils`, a pip-installable Python package that replaces the four slowest parts of a pure-Python object tracker with C++ implementations. This is where every lesson comes together.
- Slide 2: The four components -- FastKalmanFilter (pre-allocated matrices, in-place updates from L5/L10), FastPreprocessor (fused C++ kernel from L7), FastHistoryBuffer (circular buffer with zero-copy views from L4), FastStateMachine (std::variant dispatch from L8).
- Slide 3: Acceptance criteria -- All four components callable from Python, `pip install .` works, each component achieves >2x speedup over baseline, all tests pass.
- Slide 4: Grading rubric -- Correctness 30%, Performance 30%, Code quality 20%, Packaging 10%, Documentation 10%. Self-assessment after completion.
- Slide 5: Jetson capstone note -- If working on Jetson, the FastPreprocessor should use CUDA unified memory instead of explicit transfers. Test on both desktop GPU and Jetson if available. Benchmark results will differ due to unified memory architecture.

### Live Demo
- Run the baseline tracker (`tracker_baseline.py`) and the baseline benchmark (`benchmark_baseline.py`). Show the per-component timing breakdown and identify which bottleneck is worst.

### Key Takeaway
- The capstone is a realistic integration exercise -- you will apply profiling (L10), nanobind (L4), pre-allocation (L5), fused kernels (L7), and variant state machines (L8) to build a production-ready package.

## Video 12.2: Implementing FastHistoryBuffer and FastPreprocessor (~12 min)

### Slides
- Slide 1: Recommended implementation order -- Start with FastHistoryBuffer (simplest C++ surface area), then FastPreprocessor (one fused kernel, easy to validate), then FastKalmanFilter (linear algebra), finally FastStateMachine (variant + visit).
- Slide 2: FastHistoryBuffer approach -- Circular buffer in C++ with a fixed-capacity array. `push()` writes to the next slot. `latest(n)` returns an nb::ndarray view into the buffer memory. Zero-copy: the caller sees a numpy array backed by C++ storage.
- Slide 3: FastHistoryBuffer key techniques -- nb::ndarray from Lesson 4 for zero-copy views. Pre-allocated storage from Lesson 5. Circular indexing with modular arithmetic.
- Slide 4: FastPreprocessor approach -- Single C++ function that takes uint8 HWC input and produces float32 CHW normalized output. On desktop GPU: fused CUDA kernel from Lesson 7. CPU fallback: single-pass C++ loop using std::execution::unseq from Lesson 1.
- Slide 5: Testing each component -- Write a Python test that creates both the baseline and fast version, feeds identical inputs, and verifies outputs match within tolerance. Then benchmark both to confirm >2x speedup.

### Key Takeaway
- Start with the simplest component to build confidence, then tackle increasingly complex ones -- always test correctness before measuring performance.

## Video 12.3: Implementing FastKalmanFilter and FastStateMachine (~12 min)

### Slides
- Slide 1: FastKalmanFilter approach -- Pre-allocate all matrices (F, P, Q, state) in the constructor. predict() and update() operate in-place with no per-frame allocations. Use raw buffer math or a lightweight linear algebra approach.
- Slide 2: FastKalmanFilter key techniques -- Pre-allocation from Lessons 5 and 10. In-place matrix operations. nb::ndarray for returning state to Python. The fix from Lesson 10 Round 1 (moving np.eye to __init__) implemented in C++.
- Slide 3: FastStateMachine approach -- Define states as structs: `Idle{}`, `Tracking{BBox target}`, `Lost{int frames_lost}`, `Search{BBox last_known}`. Use `std::variant<Idle, Tracking, Lost, Search>` with `std::visit` for transitions. From Lesson 8.
- Slide 4: FastStateMachine nanobind bindings -- Expose state names as strings to Python for display, but use variant dispatch internally. `get_state_name()` returns a string, `transition()` takes a detection optional and returns the new state.
- Slide 5: Integration testing -- Run both baseline and fast pipelines on the same sequence of frames. Verify state transitions match. Verify detection outputs match. Then compare total pipeline time.

### Key Takeaway
- The FastKalmanFilter and FastStateMachine demonstrate how C++ compile-time techniques (pre-allocation, variant dispatch) replace Python runtime overhead with zero-cost abstractions.

## Video 12.4: Packaging and Final Benchmark (~10 min)

### Slides
- Slide 1: Package structure -- pyproject.toml with scikit-build-core, CMakeLists.txt building all four components, src/fast_tracker_utils/__init__.py importing all modules. From Lesson 9.
- Slide 2: Building and installing -- `pip install .` should compile all C++ code and install the package. Verify with `python3 -c "from fast_tracker_utils import FastKalmanFilter; print('OK')"`.
- Slide 3: Final benchmark -- Run `benchmark_fast.py` for side-by-side comparison. Expected results: each component >2x faster, full pipeline >3x faster. Show the before/after table.
- Slide 4: Self-assessment -- Score yourself on correctness (tests pass), performance (>2x per component), code quality (modern C++, no leaks), packaging (pip install works), documentation (docstrings, benchmark results).
- Slide 5: Jetson final note -- If deploying on Jetson, run the benchmark on both desktop and Jetson. Note the differences in absolute timing (Jetson is slower) but similar relative speedups. Unified memory may change which optimizations have the biggest impact.

### Live Demo
- Run `pip install .` on the completed capstone project, execute the final benchmark, and walk through the results comparing baseline vs fast for all four components.

### Key Takeaway
- The capstone proves you can take a real Python codebase, profile its bottlenecks, implement targeted C++ replacements, and ship the result as a pip-installable package.
