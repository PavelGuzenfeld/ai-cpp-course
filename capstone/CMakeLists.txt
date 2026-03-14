cmake_minimum_required(VERSION 3.15)
project(fast_tracker_utils LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Dependencies ----------------------------------------------------------
find_package(pybind11 REQUIRED)

# --- Native extension module -----------------------------------------------
# Trainees: add your C++ source files here.  Each component gets its own
# source file under src/ and is compiled into a single _native extension.
#
# Example (uncomment and adapt as you implement each component):
#
# pybind11_add_module(_native
#     src/kalman_native.cpp
#     src/preprocess_native.cpp
#     src/history_native.cpp
#     src/state_machine_native.cpp
#     src/bindings.cpp
# )
#
# target_include_directories(_native PRIVATE src)
#
# install(TARGETS _native
#     LIBRARY DESTINATION fast_tracker_utils
# )

# --- Optional: Eigen for KalmanFilter -------------------------------------
# If you want to use Eigen for the Kalman filter implementation:
#
# find_package(Eigen3 3.3 REQUIRED NO_MODULE)
# target_link_libraries(_native PRIVATE Eigen3::Eigen)
