# Section 9 Quiz: Going to Production -- Packaging Your Work

## Q1: What problem does `pip install .` solve that `colcon build && source install/setup.bash && export PYTHONPATH=...` does not?

- a) It compiles code faster
- b) It handles finding Python, compiling extensions, installing to the correct location, and managing dependencies automatically, eliminating fragile manual environment setup
- c) It supports more programming languages
- d) It makes the code run faster

**Answer: b)** Proper Python packaging through `pip install` automates compilation, installation, and dependency resolution. Manual `PYTHONPATH` manipulation, `source setup.bash`, and `LD_LIBRARY_PATH` exports are fragile, error-prone, and break when any path changes.

## Q2: Why does scikit-build-core exist when setuptools already supports C extensions?

- a) scikit-build-core is older and more stable
- b) scikit-build-core delegates compilation to CMake, allowing you to reuse existing CMakeLists.txt files with full CMake features like `find_package`, instead of manually specifying every compiler flag in setuptools
- c) setuptools cannot compile C code at all
- d) scikit-build-core supports Python 2

**Answer: b)** setuptools requires manually specifying include paths, compiler flags, and libraries for each extension. scikit-build-core bridges CMake and Python packaging, so your existing CMakeLists.txt handles all compilation details, including nanobind integration, TBB, and OpenCV dependencies.

## Q3: What is the purpose of the `src` layout in Python packaging?

- a) It makes the source code harder to find
- b) It prevents accidentally importing the local source directory instead of the installed package, catching packaging errors early
- c) It is required by the Python language specification
- d) It improves runtime performance

**Answer: b)** Without `src` layout, running `python -c "import my_package"` from the project root imports the local source directory, masking packaging errors. With `src` layout, the package is only importable after proper installation, ensuring your packaging configuration is correct.

## Q4: Why should the package version come from a single `VERSION` file rather than being hardcoded in multiple places?

- a) Multiple files are harder to read
- b) A single source of truth prevents version drift between CMakeLists.txt, pyproject.toml, and Python runtime, which causes confusing bugs
- c) Git does not support versioning
- d) Python cannot parse version strings

**Answer: b)** If the version is hardcoded in pyproject.toml, CMakeLists.txt, and `__init__.py`, a developer may update one but forget the others. A single `VERSION` file read by all three consumers guarantees they always agree.

## Q5: What do `.pyi` type stubs and the `py.typed` marker file provide for C++ extensions?

- a) Runtime type checking that prevents crashes
- b) IDE autocomplete, type checking support from tools like mypy, and documentation of the C++ API's Python interface
- c) Automatic conversion between C++ and Python types
- d) Faster function calls by pre-resolving types

**Answer: b)** C++ extensions are opaque to Python tooling. Without `.pyi` stubs, IDEs show no autocomplete, and type checkers cannot validate usage. The `py.typed` marker (PEP 561) tells tools that the package ships type information.

## Q6: In a multi-stage Docker build for a C++ Python extension, why is the runtime image so much smaller than the builder image?

- a) The runtime image uses a different operating system
- b) The builder contains compilers, headers, CMake, and development libraries (3+ GB) that are not needed to run the compiled extension; the runtime image only needs Python and the installed wheel
- c) The runtime image compresses all files
- d) Docker automatically removes unused layers

**Answer: b)** The build stage needs g++, cmake, development headers, CUDA toolkit, etc., which total several gigabytes. The runtime stage only needs the Python interpreter and the compiled `.so` file from the wheel, typically 200-400 MB total.

## Q7: What does the wheel filename `tracker_utils-0.1.0-cp310-cp310-linux_x86_64.whl` encode?

- a) Just the package name and version
- b) Package name, version, Python version (CPython 3.10), ABI tag, and platform (Linux x86_64) -- indicating this is a platform-specific binary wheel
- c) The Git commit hash and branch name
- d) The number of files in the package

**Answer: b)** Each component specifies compatibility: `cp310` means CPython 3.10, and `linux_x86_64` means it contains compiled code for that specific platform. You need to build separate wheels for each Python version and OS/architecture combination.

## Q8: What is the role of `[project.scripts]` in pyproject.toml?

- a) It defines Python scripts that run during installation
- b) It declares entry points that pip installs as command-line tools, mapping a command name to a Python function so users can run `tracker-bbox` directly instead of `python3 scripts/run_tracker.py`
- c) It lists all source files in the project
- d) It configures the test runner

**Answer: b)** `[project.scripts]` creates console entry points. After `pip install`, the specified commands become available system-wide, replacing the pattern of running scripts through `python3 path/to/script.py` with proper installed commands.
