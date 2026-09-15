cmake_minimum_required(VERSION 3.15)
project(fast_tracker_utils LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Dependencies ----------------------------------------------------------
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# nanobind ships its own CMake package; scikit-build-core installs it into
# the isolated build environment, so locate it via the interpreter rather
# than a hardcoded path (unlike the Docker-only lessons elsewhere in this
# course, this package must also build in a bare `pip install .` venv).
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# --- Four native extension modules, one per capstone component ------------
# NOMINSIZE on the two with an actual per-pixel/per-frame compute kernel:
# nanobind_add_module defaults to -Os for typical thin-wrapper bindings,
# which silently overrides -O3 and was costing preprocess_native a real
# 2x -- these two are hot loops, not glue code.
nanobind_add_module(kalman_native NOMINSIZE src/kalman_native.cpp)
nanobind_add_module(preprocess_native NOMINSIZE src/preprocess_native.cpp)
nanobind_add_module(history_native src/history_native.cpp)
nanobind_add_module(state_machine_native src/state_machine_native.cpp)

install(TARGETS kalman_native preprocess_native history_native state_machine_native
    LIBRARY DESTINATION fast_tracker_utils)
