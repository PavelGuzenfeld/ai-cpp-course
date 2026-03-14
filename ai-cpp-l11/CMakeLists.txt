cmake_minimum_required(VERSION 3.20)
project(nanobind-l11)

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

# --- Sanitizer option for debug builds ---
option(ENABLE_SANITIZERS "Enable ASAN and UBSAN" OFF)

if(ENABLE_SANITIZERS)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address,undefined)
    message(STATUS "AddressSanitizer and UndefinedBehaviorSanitizer enabled")
endif()

# --- safe_views module ---
nanobind_add_module(safe_views NB_STATIC safe_views.cpp)
set_target_properties(safe_views PROPERTIES PREFIX "" SUFFIX ".so")
install(TARGETS safe_views
    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)

# --- memory_safety_demo module ---
nanobind_add_module(memory_safety_demo NB_STATIC memory_safety_demo.cpp)
set_target_properties(memory_safety_demo PROPERTIES PREFIX "" SUFFIX ".so")
install(TARGETS memory_safety_demo
    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)

# --- asan_example standalone executable (not a Python module) ---
add_executable(asan_example asan_example.cpp)
set_target_properties(asan_example PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON)

# asan_example always gets sanitizers when ENABLE_SANITIZERS is on
if(ENABLE_SANITIZERS)
    target_compile_options(asan_example PRIVATE
        -fsanitize=address,undefined -fno-omit-frame-pointer -g -O0)
    target_link_options(asan_example PRIVATE
        -fsanitize=address,undefined)
endif()
