cmake_minimum_required(VERSION 3.20)
project(nanobind-l17)

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

# --- Falsifier module: std::sort vs. insertion sort, timed ---
nanobind_add_module(falsifier_native NB_STATIC falsifier_native.cpp)
set_target_properties(falsifier_native PROPERTIES PREFIX "" SUFFIX ".so")
install(TARGETS falsifier_native
    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)
