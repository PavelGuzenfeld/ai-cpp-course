cmake_minimum_required(VERSION 3.20)
project(hw_latency_measurement)

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

# Set compilation options
function(set_perf_compile_options target)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${target} PRIVATE
            -O3                # Max optimization
            -march=native      # Enable all CPU-specific optimizations
            -funroll-loops     # Unroll loops for speed
            -ffast-math        # Allow unsafe but faster math optimizations
        )
        target_link_options(${target} PRIVATE
            -flto              # Enable link-time optimization
        )
    elseif(MSVC)
        target_compile_options(${target} PRIVATE
            /O2                # Max optimization
            /GL                # Whole program optimization
        )
        target_link_options(${target} PRIVATE
            /LTCG              # Link-time code generation
        )
    endif()
endfunction()

# Latency timer module
nanobind_add_module(latency_timer NB_STATIC latency_timer.cpp)
set_perf_compile_options(latency_timer)
set_target_properties(latency_timer PROPERTIES PREFIX "" SUFFIX ".so")
install(TARGETS latency_timer
    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)

# Cache benchmark module
nanobind_add_module(cache_benchmark NB_STATIC cache_benchmark.cpp)
set_perf_compile_options(cache_benchmark)
set_target_properties(cache_benchmark PROPERTIES PREFIX "" SUFFIX ".so")
install(TARGETS cache_benchmark
    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)
