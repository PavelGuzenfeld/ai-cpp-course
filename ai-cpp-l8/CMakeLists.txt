cmake_minimum_required(VERSION 3.20)
project(nanobind-l8)

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

# --- Concepts Demo module ---
nanobind_add_module(concepts_demo NB_STATIC concepts_demo.cpp)
set_target_properties(concepts_demo PROPERTIES PREFIX "" SUFFIX ".so")
install(TARGETS concepts_demo
    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)

# --- Compile-Time LUT module ---
nanobind_add_module(compile_time_lut NB_STATIC compile_time_lut.cpp)
set_target_properties(compile_time_lut PROPERTIES PREFIX "" SUFFIX ".so")
install(TARGETS compile_time_lut
    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)

# --- State Machine module ---
nanobind_add_module(state_machine NB_STATIC state_machine.cpp)
set_target_properties(state_machine PROPERTIES PREFIX "" SUFFIX ".so")
install(TARGETS state_machine
    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)
