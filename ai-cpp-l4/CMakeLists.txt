cmake_minimum_required(VERSION 3.20)
project(nanobind-l4)

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

# --- BoundingBox module ---
nanobind_add_module(bbox_native NB_STATIC bbox_native.cpp)
set_target_properties(bbox_native PROPERTIES PREFIX "" SUFFIX ".so")
install(TARGETS bbox_native
    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)

# --- Buffer Pool module ---
nanobind_add_module(buffer_pool_native NB_STATIC buffer_pool_native.cpp)
set_target_properties(buffer_pool_native PROPERTIES PREFIX "" SUFFIX ".so")
install(TARGETS buffer_pool_native
    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)

# --- History View module ---
nanobind_add_module(history_view_native NB_STATIC history_view_native.cpp)
set_target_properties(history_view_native PROPERTIES PREFIX "" SUFFIX ".so")
install(TARGETS history_view_native
    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)
