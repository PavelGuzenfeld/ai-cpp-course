# Assignment 4: Capstone — fast_tracker_utils

## Objective

Build a pip-installable Python package that replaces the four slowest parts of
a pure-Python object tracker with C++ implementations. This is the final project
that demonstrates mastery of the entire course.

## Background

See `capstone/README.md` for the full project description, grading rubric, and
step-by-step guide. The capstone ties together every lesson:

| Component | Lessons Used |
|-----------|-------------|
| FastKalmanFilter | L1 (SIMD), L4 (nanobind), L5 (pre-allocation) |
| FastPreprocessor | L2 (cache), L7 (CUDA kernels) |
| FastHistoryBuffer | L4 (zero-copy views), L3 (shared memory patterns) |
| FastStateMachine | L8 (compile-time concepts, variant) |

## Jetson Track (Optional, +25 bonus points)

If targeting Jetson deployment, additionally:

1. **Cross-compile or native build** the package on Jetson using `Dockerfile.jetson`
2. **Use unified memory** in FastPreprocessor instead of explicit copies
3. **Benchmark on Jetson** in both MAX and 15W power modes
4. **Create a `Dockerfile.tracker`** that packages the complete tracker as a
   deployable container for Jetson Orin

### Jetson Acceptance Criteria

- [ ] Package installs on JetPack 6.x
- [ ] All tests pass on Jetson
- [ ] Benchmark shows >2x improvement over baseline on Jetson
- [ ] Power consumption documented (using `tegrastats`)

## Deliverables

### Standard Track
- Working `fast_tracker_utils` package (see `capstone/README.md`)
- Benchmark results showing >2x per component, >3x pipeline
- All tests passing

### Jetson Track (bonus)
- Jetson benchmark results
- Power analysis with `tegrastats`
- `Dockerfile.tracker` for Jetson deployment

## Grading

See `capstone/README.md` for the standard 100-point rubric.
Jetson track adds up to 25 bonus points.
