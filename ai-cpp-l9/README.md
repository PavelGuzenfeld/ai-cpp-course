# Lesson 9: Going to Production — Packaging Your Work

## Goal

Package a C++ extension as a pip-installable Python package. By the end of this lesson,
you will have a real `tracker-utils` package that anyone can install with `pip install .`,
complete with type stubs, tests, and a production Docker image.

## Why Packaging Matters

Every lesson so far has followed the same pattern: build with CMake or colcon, set
`PYTHONPATH`, and hope for the best. That works in development. It falls apart in
production.

The [tracker_engine](https://github.com/thebandofficial/tracker_engine) codebase
demonstrates this problem clearly:

```bash
# Current tracker_engine workflow
cd tracker_engine
colcon build
source install/setup.bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/trt
python3 run_tracker.py
```

Every developer must repeat these steps. Every deployment script must replicate them.
The `trt/` directory contains TensorRT wrappers that *should* be compiled extensions
but are instead pure Python files that call into shared libraries manually — fragile
paths, manual `LD_LIBRARY_PATH` manipulation, and no version tracking.

Compare that to what packaging gives you:

```bash
pip install tracker-utils
python3 -c "from tracker_utils import BBox; print(BBox(0, 0, 100, 50).area)"
```

No `PYTHONPATH`. No `source install/setup.bash`. No "did you remember to colcon build?"
The C++ extension compiles automatically during `pip install`, ends up in the right
location, and just works.

### What breaks without proper packaging:

| Problem | Symptom |
|---------|---------|
| Missing `PYTHONPATH` | `ModuleNotFoundError` |
| Wrong Python version | Segfault or `ImportError: undefined symbol` |
| Missing shared libraries | `OSError: libfoo.so: cannot open shared object file` |
| No version tracking | "It worked on my machine" |
| No dependency declaration | Manual install steps in README |

## Build and Run

```bash
# Inside Docker container
cd /workspace/ai-cpp-l9

# Install the package in development mode
pip install -e .

# Run tests
pytest test_package.py test_integration_package.py -v

# Build a distributable wheel
pip wheel . -w dist/

# Inspect the wheel contents
unzip -l dist/tracker_utils-*.whl

# Full build + test cycle
bash build_and_test.sh
```

## The Python Packaging Landscape

Before diving into our package, let's survey the tools available. If you've only
used `pip install numpy`, the packaging ecosystem can feel overwhelming. Here's
the map.

### setuptools (legacy, still everywhere)

The original packaging tool. Most Python packages you've ever used were built with
setuptools. It works fine for pure Python packages but struggles with C++ extensions:

```python
# setup.py — the old way
from setuptools import setup, Extension

setup(
    ext_modules=[
        Extension("bbox_native", sources=["bbox_native.cpp"],
                  include_dirs=["/usr/local/include"],
                  extra_compile_args=["-std=c++23"])
    ]
)
```

Problems: No CMake support. You must specify every compiler flag, include path, and
library manually. Nanobind's CMake integration is completely bypassed.

tracker_engine uses setuptools with a `pyproject.toml` that declares basic metadata
but doesn't handle compiled extensions at all — the C++ parts live outside the
package system entirely.

### [scikit-build-core](https://github.com/scikit-build/scikit-build-core) (modern CMake integration)

scikit-build-core bridges the gap between CMake and Python packaging. It replaces
setuptools as the build backend and delegates all compilation to CMake:

```toml
[build-system]
requires = ["scikit-build-core", "nanobind"]
build-backend = "scikit_build_core.build"
```

Your existing `CMakeLists.txt` does the actual compilation. scikit-build-core handles
the packaging part: finding Python, setting install paths, and creating wheels.

This is what we use in this lesson.

### [meson-python](https://github.com/mesonbuild/meson-python)

An alternative to scikit-build-core that uses [Meson](https://mesonbuild.com/) instead of CMake. Popular in the
scientific Python ecosystem (numpy, scipy use it). If your project already uses Meson,
meson-python is the natural choice. Otherwise, scikit-build-core is simpler for
CMake-based projects.

### [nanobind](https://github.com/wjakob/nanobind)'s built-in scikit-build support

Nanobind was designed to work with scikit-build-core from day one. The
`nanobind_add_module()` CMake function automatically handles:
- Finding the correct Python interpreter
- Setting the output directory for scikit-build
- Configuring platform-specific extension suffixes (`.cpython-310-x86_64-linux-gnu.so`)
- Static linking the nanobind runtime (`NB_STATIC`)

This tight integration is why scikit-build-core + nanobind is the recommended stack.

## pyproject.toml Anatomy

The `pyproject.toml` file is the single source of truth for your package. Let's
dissect each section:

### [build-system] — How to build

```toml
[build-system]
requires = ["scikit-build-core>=0.10", "nanobind>=2.0"]
build-backend = "scikit_build_core.build"
```

- `requires`: Packages pip must install *before* building your package. These are
  build-time dependencies, not runtime dependencies.
- `build-backend`: The Python entry point that pip calls to build your package.
  This replaces the old `python setup.py build` mechanism.

### [project] — What you're shipping

```toml
[project]
name = "tracker-utils"
version = "0.1.0"
description = "C++ accelerated tracking utilities"
requires-python = ">=3.10"
dependencies = ["numpy>=1.24"]
```

- `name`: The package name on PyPI (hyphens are conventional; underscores in import names).
- `version`: Semantic version. We'll read this from a `VERSION` file.
- `requires-python`: Minimum Python version. C++23 nanobind modules need 3.10+.
- `dependencies`: Runtime dependencies. These get installed when someone `pip install`s
  your package.

### [project.optional-dependencies] — Extras

```toml
[project.optional-dependencies]
test = ["pytest>=7.0", "numpy>=1.24"]
```

Install with `pip install .[test]`. This keeps test dependencies out of production.

### [tool.scikit-build] — Build configuration

```toml
[tool.scikit-build]
wheel.packages = ["src/tracker_utils"]
cmake.build-type = "Release"
```

Tells scikit-build-core where to find your Python package and what CMake build type
to use. The `wheel.packages` directive is critical — it maps your source layout to
the wheel's internal structure.

## Packaging a Nanobind Extension

### Step 1: Project Structure

We use the `src` layout, which is the modern Python packaging convention:

```
ai-cpp-l9/
  pyproject.toml          # Package configuration
  CMakeLists.txt          # C++ build rules
  VERSION                 # Single source of truth for version
  src/
    tracker_utils/
      __init__.py         # Package init, imports native extension
      bbox.py             # High-level Python API
      _native.cpp         # C++ extension source (nanobind)
      _native.pyi         # Type stubs for IDE autocomplete
      py.typed            # PEP 561 marker: "this package has types"
  test_package.py         # Unit tests
  test_integration_package.py  # Integration tests
  Dockerfile.prod         # Production Docker image
  build_and_test.sh       # Build + test script
```

Why `src` layout? It prevents a common mistake: importing your package from the source
directory instead of the installed version. With `src` layout, Python *cannot* find
`tracker_utils` unless it's actually installed, which catches packaging errors early.

### Step 2: pyproject.toml with scikit-build-core

See `pyproject.toml` in this lesson directory for a complete, working example.
The key parts:

1. `build-system` declares scikit-build-core and nanobind as build dependencies
2. `project` declares the package metadata and runtime dependencies
3. `tool.scikit-build` configures the wheel layout and CMake options
4. Version is read dynamically from a `VERSION` file

### Step 3: CMakeLists.txt Integration

The CMakeLists.txt must do two things differently from our L4 version:

1. **Use `install()`**: scikit-build-core picks up installed targets automatically
2. **Use the `SKBUILD` marker**: Detect when building inside scikit-build vs standalone

```cmake
if(DEFINED SKBUILD)
    # scikit-build-core sets this — install into the Python package
    install(TARGETS _native DESTINATION tracker_utils)
else()
    # Standalone CMake build — install to standard location
    install(TARGETS _native DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
```

The `SKBUILD` variable is set automatically by scikit-build-core. When it's defined,
your `install(TARGETS ...)` destinations are relative to the Python package directory,
not the system prefix. This is what makes `pip install .` put the `.so` in the right
place. See also [L4](../ai-cpp-l4/) for nanobind fundamentals.

### Step 4: Version Management

Never hardcode versions in multiple places. Use a single `VERSION` file:

```
0.1.0
```

Read it in CMakeLists.txt:
```cmake
file(READ "${CMAKE_SOURCE_DIR}/VERSION" PROJECT_VERSION)
string(STRIP "${PROJECT_VERSION}" PROJECT_VERSION)
```

Read it in pyproject.toml via scikit-build-core:
```toml
[project]
dynamic = ["version"]

[tool.scikit-build.metadata.version]
provider = "scikit_build_core.metadata.regex"
input = "VERSION"
```

Read it at runtime in Python:
```python
from importlib.metadata import version
__version__ = version("tracker-utils")
```

One file, three consumers, zero drift.

### Step 5: Building Wheels

A "wheel" is a `.whl` file — a zip archive with a specific layout that pip can install
without building. It's the standard distribution format for Python packages.

```bash
# Install directly (builds + installs)
pip install .

# Build a wheel (for distribution)
pip wheel . -w dist/

# Install the wheel on another machine
pip install dist/tracker_utils-0.1.0-cp310-cp310-linux_x86_64.whl
```

The wheel filename encodes:
- Package name and version (`tracker_utils-0.1.0`)
- Python version (`cp310`)
- Platform (`linux_x86_64`)

Since our package contains a compiled C++ extension, the wheel is platform-specific.
You need to build separate wheels for each OS/architecture combination.

## Including Data Files

Real packages often need to ship data: test images, configuration YAML files, model
weights, calibration parameters. There are two mechanisms:

### Package data (small files that ship with the code)

```toml
[tool.scikit-build]
wheel.packages = ["src/tracker_utils"]

# In your package directory, just include the files:
# src/tracker_utils/data/default_config.yaml
# src/tracker_utils/data/calibration.json
```

Access at runtime:
```python
from importlib.resources import files
config_path = files("tracker_utils").joinpath("data/default_config.yaml")
```

### External data (large files downloaded on demand)

Model weights and large datasets should not be in your wheel. Instead:
```python
def download_model(name: str, cache_dir: str = "~/.tracker_utils/models"):
    """Download model weights on first use, cache locally."""
    ...
```

## Entry Points and CLI Tools

You can expose Python functions as command-line tools:

```toml
[project.scripts]
tracker-bbox = "tracker_utils.cli:main"
```

After `pip install .`, the command `tracker-bbox` is available system-wide:

```bash
$ tracker-bbox --help
Usage: tracker-bbox [OPTIONS] IMAGE
  Detect and display bounding boxes in an image.
```

This replaces the tracker_engine pattern of `python3 scripts/run_tracker.py` with a
proper installed command.

## Type Stubs for C++ Extensions

C++ extensions are opaque to Python type checkers and IDEs. Without type stubs,
you get no autocomplete, no type checking, and red squiggles everywhere.

A `.pyi` file (type stub) declares the types without any implementation:

```python
# _native.pyi
class BBox:
    x: float
    y: float
    w: float
    h: float

    def __init__(self, x: float, y: float, w: float, h: float) -> None: ...
    @property
    def area(self) -> float: ...
    def iou(self, other: BBox) -> float: ...
```

The `py.typed` marker file (PEP 561) tells tools like mypy and pyright that your
package includes type information. Without it, type checkers ignore your stubs.

## Docker for Distribution

Production deployments need reproducible environments. A multi-stage Docker build
separates the build environment (compilers, headers, CMake) from the runtime
environment (just Python + your wheel).

```dockerfile
# Stage 1: Build
FROM ai-cpp-course AS builder
COPY . /build
RUN cd /build && pip wheel . -w /wheels

# Stage 2: Runtime
FROM python:3.10-slim
COPY --from=builder /wheels /wheels
RUN pip install /wheels/*.whl && rm -rf /wheels
```

The builder stage might be 3+ GB (compilers, CUDA, development headers). The runtime
stage is typically 200-400 MB — just Python and your compiled extension.

See `Dockerfile.prod` in this lesson for a complete example.

## Continuous Integration

Your CI pipeline should:

1. **Build the wheel** on every push
2. **Run tests** against the installed package (not the source tree)
3. **Build for multiple platforms** if you distribute publicly

```yaml
# .github/workflows/build.yml (simplified)
jobs:
  build:
    runs-on: ubuntu-latest
    container: ai-cpp-course:latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install .[test]
      - run: pytest --tb=short
      - run: pip wheel . -w dist/
      - uses: actions/upload-artifact@v4
        with:
          name: wheels
          path: dist/*.whl
```

For cross-platform wheel builds, use [cibuildwheel](https://cibuildwheel.pypa.io/)
which automates building wheels for Linux, macOS, and Windows across multiple Python
versions.

## Versioning Strategy

### Semantic Versioning (SemVer)

```
MAJOR.MINOR.PATCH
  |     |     |
  |     |     +-- Bug fixes, no API change
  |     +-------- New features, backwards compatible
  +-------------- Breaking API changes
```

For a tracking library:
- `0.1.0` -> `0.1.1`: Fixed IoU calculation edge case
- `0.1.1` -> `0.2.0`: Added `BBox.scale()` method
- `0.2.0` -> `1.0.0`: Changed `BBox` constructor signature

### Automated version from git tags

For mature projects, derive the version from git tags:

```toml
[tool.scikit-build.metadata.version]
provider = "scikit_build_core.metadata.setuptools_scm"

[tool.setuptools_scm]
```

This reads the version from `git describe`, so `git tag v1.2.3` automatically sets
the package version. No file to edit, no version drift.

For this lesson, we use the simpler `VERSION` file approach.

## tracker_engine: What Would Change

Let's look at how tracker_engine's packaging would transform with proper Python
packaging.

### Before (current state)

```
tracker_engine/
  pyproject.toml          # Basic metadata, no C++ build
  setup.py                # Legacy, mostly empty
  trt/
    trt_inference.py      # Pure Python calling ctypes/cffi to .so files
    libtrt_wrapper.so     # Manually compiled, manually placed
  scripts/
    run_tracker.py        # python3 scripts/run_tracker.py
  install.sh              # Manual build steps
```

Deployment requires:
```bash
git clone tracker_engine
cd tracker_engine
bash install.sh           # Compile C++, copy .so files
colcon build              # Build ROS/colcon packages
source install/setup.bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/trt
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/tensorrt/lib
python3 scripts/run_tracker.py
```

### After (properly packaged)

```
tracker_engine/
  pyproject.toml          # Full config with scikit-build-core
  CMakeLists.txt          # Builds TRT wrappers as nanobind extensions
  VERSION                 # 1.0.0
  src/
    tracker_engine/
      __init__.py
      trt/
        __init__.py
        _trt_native.cpp   # TensorRT wrapper as nanobind module
        _trt_native.pyi   # Type stubs
        inference.py      # High-level Python API
      py.typed
```

Deployment becomes:
```bash
pip install tracker-engine
tracker-run --model yolov8 --input camera0
```

The `trt/` directory's manual shared library loading is replaced by a proper nanobind
extension. The `scripts/` directory's entry points become `[project.scripts]` console
commands. The colcon/PYTHONPATH dance disappears entirely.

## Exercises

### Exercise 1: Build and Install
Build the `tracker-utils` package and verify it works:
```bash
cd /workspace/ai-cpp-l9
pip install .
python3 -c "from tracker_utils import BBox; b = BBox(10, 20, 100, 50); print(b)"
```
Verify that the C++ extension is loaded (not a pure Python fallback).

### Exercise 2: Inspect the Wheel
Build a wheel and examine its contents:
```bash
pip wheel . -w dist/
unzip -l dist/tracker_utils-*.whl
```
What files are inside? What is the platform tag? Why is the wheel platform-specific?

### Exercise 3: Add a CLI Entry Point
Add a `tracker-iou` console script that takes four bbox coordinates on the command line
and computes IoU against a second bbox. Update `pyproject.toml` with the entry point and
create the CLI module. Verify it works after reinstalling:
```bash
pip install .
tracker-iou 0 0 100 100 50 50 100 100
# Should print: IoU = 0.142857...
```

### Exercise 4: Add Package Data
Create a `data/` directory inside `src/tracker_utils/` with a sample YAML config file.
Use `importlib.resources` to load it at runtime. Verify the file is included in the
wheel.

### Exercise 5: Production Docker Image
Build the production Docker image and compare its size to the development image:
```bash
docker build -f Dockerfile.prod -t tracker-utils-prod .
docker images | grep tracker
```
Run the tests inside the production container to verify everything works without
the development toolchain.

## What You Learned

- Python packaging replaces manual `PYTHONPATH`/colcon workflows with `pip install`
- scikit-build-core bridges CMake and Python packaging, letting you reuse existing CMakeLists.txt
- `pyproject.toml` is the single configuration file for metadata, dependencies, and build settings
- The `src` layout prevents accidentally importing uninstalled code
- Version should come from a single source (VERSION file or git tags), never hardcoded in multiple places
- Type stubs (`.pyi`) and `py.typed` give IDE autocomplete for C++ extensions
- Multi-stage Docker builds separate 3+ GB build environments from slim runtime images
- Wheels are the standard distribution format for Python packages with compiled extensions

## Lesson Files

| File | Description |
|------|-------------|
| [CMakeLists.txt](CMakeLists.txt) | CMake build rules for nanobind extension |
| [pyproject.toml](pyproject.toml) | Package configuration with scikit-build-core |
| [VERSION](VERSION) | Single source of truth for package version |
| [Dockerfile.prod](Dockerfile.prod) | Multi-stage production Docker image |
| [build_and_test.sh](build_and_test.sh) | Build, test, and wheel creation script |
| [test_package.py](test_package.py) | Unit tests for the package |
| [test_integration_package.py](test_integration_package.py) | Integration tests for installed package |
