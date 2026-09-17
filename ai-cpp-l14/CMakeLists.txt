cmake_minimum_required(VERSION 3.20)
project(nanobind-l14)

# Require C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find dependencies
find_package(Python3 COMPONENTS Interpreter Development)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(nanobind CONFIG REQUIRED HINTS /usr/local/nanobind/cmake)

# Silence warnings in nanobind headers by marking as SYSTEM include
include_directories(SYSTEM /usr/local/nanobind/include)

# --- Mock device module: what CI actually builds and tests ---
nanobind_add_module(device_mock_native NB_STATIC device_mock_native.cpp)
set_target_properties(device_mock_native PROPERTIES PREFIX "" SUFFIX ".so")
install(TARGETS device_mock_native
    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)

# --- Real device module: the same contract backed by V4L2. Built and
#     installed everywhere Linux headers exist -- it is opening /dev/video*
#     that fails without a camera, not building, which is what lets the
#     tests skip instead of stubbing. ---
nanobind_add_module(device_real_native NB_STATIC
    device_real_native.cpp real_device_v4l2.cpp)
set_target_properties(device_real_native PROPERTIES PREFIX "" SUFFIX ".so")
install(TARGETS device_real_native
    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)

# --- Real-header layout check: compiled only, never linked. Proves
#     device_buffer.hpp type-checks against the real DeviceFrame layout,
#     not just the mock's, on whatever ABI this target uses. ---
add_library(device_real_layout_check OBJECT device_real_layout_check.cpp)
set_target_properties(device_real_layout_check PROPERTIES
    CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
