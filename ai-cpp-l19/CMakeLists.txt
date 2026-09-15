cmake_minimum_required(VERSION 3.20)
project(nanobind-l19)

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

# One option controls whether module_a_native and module_b_native each get
# their own copy of linking_registry's global state, or share one copy.
option(LINKING_REGISTRY_SHARED "Build linking_registry as a shared library instead of static" OFF)

if(LINKING_REGISTRY_SHARED)
    add_library(linking_registry SHARED registry.cpp)
else()
    add_library(linking_registry STATIC registry.cpp)
endif()
target_compile_features(linking_registry PRIVATE cxx_std_23)
set_target_properties(linking_registry PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- module_a_native, module_b_native: identical, both link linking_registry ---
nanobind_add_module(module_a_native NB_STATIC module_a_native.cpp)
target_link_libraries(module_a_native PRIVATE linking_registry)
set_target_properties(module_a_native PROPERTIES PREFIX "" SUFFIX ".so"
    BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
install(TARGETS module_a_native
    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)

nanobind_add_module(module_b_native NB_STATIC module_b_native.cpp)
target_link_libraries(module_b_native PRIVATE linking_registry)
set_target_properties(module_b_native PROPERTIES PREFIX "" SUFFIX ".so"
    BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
install(TARGETS module_b_native
    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)

if(LINKING_REGISTRY_SHARED)
    install(TARGETS linking_registry
        DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)
endif()
