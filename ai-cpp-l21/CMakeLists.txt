cmake_minimum_required(VERSION 3.20)
project(nanobind-l21)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 COMPONENTS Interpreter Development)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(nanobind CONFIG REQUIRED HINTS /usr/local/nanobind/cmake)

include_directories(SYSTEM /usr/local/nanobind/include)

# NOMINSIZE: nanobind_add_module defaults to -Os, which would silently
# override the -O3 these measurements need (see ai-cpp-l23 capstone notes).
nanobind_add_module(tax_bench NB_STATIC NOMINSIZE tax_bench.cpp)
target_compile_options(tax_bench PRIVATE -O3)
set_target_properties(tax_bench PROPERTIES PREFIX "" SUFFIX ".so")
install(TARGETS tax_bench
    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)
