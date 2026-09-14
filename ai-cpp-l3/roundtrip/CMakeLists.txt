cmake_minimum_required(VERSION 3.20)
project(ai_cpp_l3)

# Require C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find dependencies
find_package(Python3 COMPONENTS Interpreter Development)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(nanobind CONFIG REQUIRED HINTS /usr/local/nanobind/cmake)
find_package(shm REQUIRED)
find_package(exception-rt REQUIRED)

# Silence warnings in nanobind headers by marking as SYSTEM include
include_directories(SYSTEM /usr/local/nanobind/include)

# --- Shared-memory round-trip module (Exercise 1) ---
nanobind_add_module(shm_roundtrip_native NB_STATIC shm_roundtrip_native.cpp)
target_include_directories(shm_roundtrip_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../flat-type/include)
target_link_libraries(shm_roundtrip_native PRIVATE shm::shm)
set_target_properties(shm_roundtrip_native PROPERTIES PREFIX "" SUFFIX ".so")
install(TARGETS shm_roundtrip_native
    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)
