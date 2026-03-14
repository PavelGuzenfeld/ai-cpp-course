cmake_minimum_required(VERSION 3.20)

# Read version from VERSION file
file(READ "${CMAKE_SOURCE_DIR}/VERSION" PROJECT_VERSION)
string(STRIP "${PROJECT_VERSION}" PROJECT_VERSION)

project(tracker_utils VERSION ${PROJECT_VERSION})

# Require C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find dependencies
find_package(Python3 COMPONENTS Interpreter Development)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(nanobind CONFIG REQUIRED HINTS /usr/local/nanobind/cmake)

# Silence warnings in nanobind headers
include_directories(SYSTEM /usr/local/nanobind/include)

# Build the nanobind extension module
nanobind_add_module(_native NB_STATIC src/tracker_utils/_native.cpp)

# Install the compiled extension into the correct location.
# When building via scikit-build-core, SKBUILD is defined and the
# destination is relative to the Python package directory.
# When building standalone with CMake, install to a standard location.
if(DEFINED SKBUILD)
    install(TARGETS _native DESTINATION tracker_utils)
else()
    install(TARGETS _native
        DESTINATION lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/tracker_utils)
endif()
