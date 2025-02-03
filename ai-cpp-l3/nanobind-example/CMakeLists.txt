cmake_minimum_required(VERSION 3.20)
project(nanobind-example)

# Minimal nanobind build example

# Require C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find dependencies
find_package(fmt REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(nanobind CONFIG REQUIRED HINTS /usr/local/nanobind/cmake)

# Silence warnings in nanobind headers by marking as SYSTEM include
include_directories(SYSTEM /usr/local/nanobind/include)

# Nanobind module for general bindings
nanobind_add_module(my_module NB_STATIC nanobind_example.cpp)
set_target_properties(my_module PROPERTIES PREFIX "" SUFFIX ".so")
install(TARGETS my_module DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)
