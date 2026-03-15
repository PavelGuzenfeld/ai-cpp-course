# Section 9: Going to Production -- Packaging Your Work

## Video 9.1: Why Packaging Matters (~8 min)

### Slides
- Slide 1: The development workflow problem -- Every lesson so far: build with CMake/colcon, set PYTHONPATH, source setup.bash, hope for the best. This works in development. It falls apart in production.
- Slide 2: tracker_engine's deployment nightmare -- 6 manual steps: git clone, bash install.sh, colcon build, source setup.bash, export PYTHONPATH, export LD_LIBRARY_PATH. Every developer repeats this. Every deployment script replicates it.
- Slide 3: What packaging gives you -- `pip install tracker-utils` then `from tracker_utils import BBox`. No PYTHONPATH. No source setup.bash. The C++ extension compiles automatically and ends up in the right location.
- Slide 4: What breaks without packaging -- Missing PYTHONPATH (ModuleNotFoundError), wrong Python version (segfault), missing shared libraries (OSError), no version tracking ("works on my machine"), no dependency declaration.
- Slide 5: Jetson packaging note -- On Jetson, proper packaging is even more critical. JetPack versions change frequently, and manual LD_LIBRARY_PATH manipulation across CUDA/cuDNN/TensorRT versions is fragile. A properly packaged wheel encapsulates all compiled extensions and their dependencies.

### Key Takeaway
- Proper Python packaging replaces fragile manual setup with `pip install` -- one command that builds, installs, and configures everything correctly.

## Video 9.2: The Python Packaging Landscape (~10 min)

### Slides
- Slide 1: setuptools (legacy) -- The original packaging tool. Works for pure Python but struggles with C++ extensions. No CMake support. Must specify every compiler flag manually.
- Slide 2: scikit-build-core (modern) -- Bridges CMake and Python packaging. Your existing CMakeLists.txt does the compilation. scikit-build-core handles finding Python, setting install paths, creating wheels. This is what we use.
- Slide 3: meson-python (alternative) -- Uses Meson instead of CMake. Popular in scientific Python (numpy, scipy). Choose based on your existing build system.
- Slide 4: nanobind's scikit-build support -- nanobind was designed for scikit-build-core from day one. `nanobind_add_module()` handles Python interpreter detection, output directory, platform-specific suffixes, static linking.
- Slide 5: pyproject.toml anatomy -- `[build-system]` declares scikit-build-core and nanobind as build dependencies. `[project]` declares metadata and runtime dependencies. `[tool.scikit-build]` configures wheel layout and CMake options.

### Key Takeaway
- scikit-build-core + nanobind is the recommended packaging stack for C++ extensions -- it reuses your existing CMakeLists.txt and handles all the Python packaging complexity.

## Video 9.3: Building a pip-Installable Package (~12 min)

### Slides
- Slide 1: Project structure (src layout) -- `pyproject.toml`, `CMakeLists.txt`, `VERSION`, `src/tracker_utils/__init__.py`, `_native.cpp`, `_native.pyi`, `py.typed`. The src layout prevents importing from source instead of the installed version.
- Slide 2: CMakeLists.txt integration -- Use `install()` for scikit-build-core. Detect `SKBUILD` variable to install into the Python package directory vs standard location.
- Slide 3: Version management -- Single `VERSION` file read by CMake (`file(READ ...)`), pyproject.toml (regex provider), and Python at runtime (`importlib.metadata.version()`). One file, three consumers, zero drift.
- Slide 4: Building wheels -- `pip install .` builds and installs. `pip wheel . -w dist/` creates a distributable .whl file. Wheel filename encodes package name, version, Python version, and platform.
- Slide 5: Type stubs for C++ extensions -- `.pyi` files declare types without implementation. `py.typed` marker (PEP 561) tells mypy/pyright the package includes type information. Essential for IDE autocomplete with C++ extensions.

### Live Demo
- Build the tracker-utils package with `pip install -e .`, run the tests, inspect the wheel contents with `unzip -l`, show IDE autocomplete working with the type stubs.

### Key Takeaway
- The src layout, single VERSION file, and type stubs create a professional-grade Python package from C++ extensions -- users get pip install, IDE autocomplete, and version tracking.

## Video 9.4: Docker Distribution and CI (~10 min)

### Slides
- Slide 1: Multi-stage Docker builds -- Stage 1 (builder): 3+ GB image with compilers, CUDA, headers. Stage 2 (runtime): 200-400 MB with just Python and compiled extensions. `COPY --from=builder` transfers only the wheel.
- Slide 2: CI pipeline -- Build wheel on every push, run tests against the installed package (not source tree), build for multiple platforms with cibuildwheel.
- Slide 3: Versioning strategy -- SemVer (MAJOR.MINOR.PATCH). Automated from git tags with setuptools_scm for mature projects. VERSION file approach for simplicity during development.
- Slide 4: Entry points and CLI tools -- `[project.scripts]` in pyproject.toml exposes Python functions as commands. `tracker-bbox --help` replaces `python3 scripts/run_tracker.py`.
- Slide 5: Jetson Docker note -- Jetson Docker images use `nvcr.io/nvidia/l4t-base` instead of standard Ubuntu. JetPack components (CUDA, cuDNN, TensorRT) must match the host JetPack version. Multi-stage builds are especially valuable on Jetson to keep deployment images small on limited storage.

### Key Takeaway
- Multi-stage Docker builds separate the 3+ GB build environment from slim runtime images -- essential for both cloud and Jetson edge deployment.
