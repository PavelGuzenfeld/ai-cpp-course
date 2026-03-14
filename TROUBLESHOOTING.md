# Common Mistakes & Troubleshooting

## Build Issues

### "Module not found" after colcon build

```
ModuleNotFoundError: No module named 'bbox_native'
```

**Fix**: You forgot to source the install:
```bash
source install/setup.bash
```
Run this every time you open a new terminal.

### "nanobind not found" during cmake

```
CMake Error: Could not find a package configuration file provided by "nanobind"
```

**Fix**: You're building outside the Docker container. All C++ builds must run
inside the Docker container:
```bash
docker run -it -v $(pwd):/workspace ai-cpp-course
```

### colcon build fails with "No such file or directory"

```
fatal error: nanobind/nanobind.h: No such file or directory
```

**Fix**: The Docker image wasn't built or is outdated:
```bash
docker build -t ai-cpp-course -f Dockerfile .
```

### "permission denied" on tests/run_all_tests.sh

```bash
chmod +x tests/run_all_tests.sh
```

## Python Issues

### torch.compile() is slower than eager mode

This is expected on the **first call**. `torch.compile()` traces and compiles
the computation graph, which takes seconds. Subsequent calls use the compiled
version. Always measure the second call onward:

```python
model = torch.compile(model)
model(warmup_input)          # slow — compiling
model(real_input)            # fast — using compiled code
```

### tracemalloc shows unexpected results

`tracemalloc` tracks Python-level allocations. It does NOT track:
- numpy's internal allocator (uses C malloc, not Python's)
- GPU memory
- Memory-mapped files

For numpy allocations, measure peak RSS instead:
```python
import resource
peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
```

### numpy views vs copies confusion

```python
a = np.array([1, 2, 3, 4])
b = a[1:3]       # view — shares memory
b[0] = 99        # modifies a[1] too!
print(a)          # [1, 99, 3, 4]
```

**Rule**: If you need the data to survive modifications to the source,
use `.copy()`. If you're just reading it in the same frame, use a view.

Check with: `np.shares_memory(a, b)`

### "Segmentation fault" when using C++ modules

The C++ module might be accessing freed memory or an out-of-bounds index.
Rebuild with AddressSanitizer:

```bash
cmake -DCMAKE_CXX_FLAGS="-fsanitize=address" ..
make
```

ASAN will print a detailed report showing exactly where the error occurred.

## GPU Issues

### "CUDA not available" on a machine with a GPU

Check if CUDA is installed and the GPU is visible:
```bash
nvidia-smi                    # should show your GPU
python3 -c "import torch; print(torch.cuda.is_available())"
```

If `nvidia-smi` works but torch doesn't see the GPU, the PyTorch version
might not match your CUDA version. Check:
```bash
python3 -c "import torch; print(torch.version.cuda)"
```

### GPU timing shows 0ms

You forgot to synchronize:
```python
# Wrong
start = time.perf_counter()
output = model(input)         # GPU hasn't finished!
elapsed = time.perf_counter() - start  # measures launch, not compute

# Right
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
output = model(input)
end.record()
torch.cuda.synchronize()     # wait for GPU to finish
ms = start.elapsed_time(end)
```

### "CUDA out of memory"

Reduce batch size, or clear the GPU cache between runs:
```python
torch.cuda.empty_cache()
```

In benchmarks, use `del tensor` to explicitly free GPU tensors before
allocating new ones.

## C++ Issues

### "undefined reference to `__cxa_allocate_exception`"

You're linking `exception-rt` but the linker order is wrong. With gcc/clang,
dependent libraries must come after the files that use them:

```
g++ my_code.o -lexception-rt   # correct
g++ -lexception-rt my_code.o   # wrong — linker hasn't seen the dependency yet
```

### ASAN reports leaks in Python C extensions

ASAN may report "leaks" that are actually Python's internal allocator holding
onto pools. Suppress these with:

```bash
ASAN_OPTIONS=detect_leaks=0 ./my_program
```

Or use a suppression file:
```bash
echo "leak:libpython" > asan_suppressions.txt
LSAN_OPTIONS=suppressions=asan_suppressions.txt ./my_program
```

### "static assertion failed: FlatType"

You tried to put a non-trivially-copyable type into shared memory:

```cpp
SharedMemory<std::string> shm("/test");  // FAILS — string has a pointer
SharedMemory<BBox> shm("/test");          // OK — BBox is a POD struct
```

Types with pointers, virtual functions, or non-trivial constructors are not
`FlatType`. Use fixed-size arrays instead of vectors/strings.

## Benchmark Issues

### My "optimized" version is slower

Common causes:
1. **Small data**: Pre-allocation overhead exceeds allocation savings for
   arrays < 100 elements. numpy's internal allocator is already fast.
2. **Warmup**: JIT-compiled code (torch.compile, CUDA kernels) is slow on
   first call. Exclude warmup from measurements.
3. **Measurement noise**: Run at least 1000 iterations and report median,
   not mean (outliers from GC or OS scheduling skew the mean).
4. **Wrong baseline**: Make sure you're comparing apples to apples. If the
   baseline includes data generation and the optimized version doesn't,
   the comparison is invalid.

### Benchmark results are inconsistent

```bash
# Disable CPU frequency scaling (requires root)
sudo cpupower frequency-set -g performance

# Disable turbo boost
echo 1 | sudo tee /sys/devices/system/cpu/intel_pturbo/no_turbo

# Pin to a specific CPU core
taskset -c 0 python3 benchmark.py
```

In Docker, you can set `--cpuset-cpus=0` to pin the container.

### "perf stat" shows zero cache misses

The hardware performance counters might not be available inside Docker.
Run on the host or use `--privileged`:

```bash
docker run --privileged -it -v $(pwd):/workspace ai-cpp-course
perf stat python3 my_benchmark.py
```

## Packaging Issues

### "pip install ." fails with scikit-build-core

Make sure you have the build dependencies:
```bash
pip install scikit-build-core nanobind
```

And that your `pyproject.toml` lists them in `[build-system] requires`:
```toml
[build-system]
requires = ["scikit-build-core", "nanobind"]
build-backend = "scikit_build_core.build"
```

### Wheel doesn't include the .so file

Check that `CMakeLists.txt` has the correct install destination:
```cmake
if(DEFINED SKBUILD)
    install(TARGETS my_module DESTINATION my_package)
endif()
```

The destination must match your Python package directory name.

### Type stubs not recognized by IDE

Make sure you have:
1. A `py.typed` marker file in your package directory
2. A `_native.pyi` stub file next to `_native.so`
3. Both files included in your package data (check `MANIFEST.in` if needed)
