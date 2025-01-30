cmake_minimum_required(VERSION 3.20)
project(flat_shared_memory)

# Require C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find dependencies
find_package(fmt REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(nanobind CONFIG REQUIRED HINTS /usr/local/nanobind/cmake)

# Silence warnings in nanobind headers by marking as SYSTEM include
include_directories(SYSTEM /usr/local/nanobind/include)

# Add an option to enable sanitizers
option(ENABLE_SANITIZERS "Enable Address/Undefined Behavior sanitizers" OFF)

# Function to apply sanitizer flags to a target
function(enable_sanitizers target)
    if(ENABLE_SANITIZERS)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_compile_options(${target} PRIVATE -fsanitize=address,undefined)
            target_link_options(${target} PRIVATE -fsanitize=address,undefined)
        elseif(MSVC)
            # Add MSVC equivalent if necessary
        endif()
    endif()
endfunction()

# Function to apply common compile and link options
function(set_common_options target)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${target} PRIVATE
            -O3                # Max optimization
            -g3                # Debugging info
            -march=native      # Enable CPU-specific optimizations
            -funroll-loops     # Unroll loops
            -fopenmp           # OpenMP for multithreading
            -ffast-math        # Fast math optimizations
            -Wstack-usage=4096 # Warn for large stack usage
            -Wall              # Enable all warnings
            -Wextra            # Enable extra warnings
            -Wpedantic         # Enable pedantic warnings
            -Werror            # Treat warnings as errors
        )
        target_link_options(${target} PRIVATE -flto) # Link-time optimization
    elseif(MSVC)
        target_compile_options(${target} PRIVATE /O2 /GL)
        target_link_options(${target} PRIVATE /LTCG)
    endif()
endfunction()

# -------------------------------
# Nanobind module using high-level API
nanobind_add_module(Share_memory_image_producer_consumer
    NB_STATIC  # Build nanobind core as static library
    flat_shm_img_producer_consumer_py.cpp
    flat_shm_producer_consumer.hpp
    image.hpp
)

target_link_libraries(Share_memory_image_producer_consumer PRIVATE pybind11::module fmt)
set_common_options(Share_memory_image_producer_consumer)
enable_sanitizers(Share_memory_image_producer_consumer)

install(TARGETS Share_memory_image_producer_consumer
    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages
)

# -------------------------------
# Nanobind module for C++ bindings (shared memory)
nanobind_add_module(Share_memory_image_producer_consumer_nb
    NB_STATIC
    flat_shm_img_producer_consumer_nb.cpp
    async_runner.cpp
    flat_shm_producer_consumer.hpp
    image.hpp
)

target_link_libraries(Share_memory_image_producer_consumer_nb PRIVATE fmt)
set_common_options(Share_memory_image_producer_consumer_nb)
enable_sanitizers(Share_memory_image_producer_consumer_nb)

install(TARGETS Share_memory_image_producer_consumer_nb DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)

# -------------------------------
# Nanobind module for general bindings
nanobind_add_module(my_module NB_STATIC nanobind_example.cpp)
set_target_properties(my_module PROPERTIES PREFIX "" SUFFIX ".so")
install(TARGETS my_module DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)

# -------------------------------
# Test executable
add_executable(flat_shm_test flat_shm_test.cpp)
target_link_libraries(flat_shm_test PRIVATE fmt)
set_common_options(flat_shm_test)
enable_sanitizers(flat_shm_test)
install(TARGETS flat_shm_test DESTINATION bin)

add_executable(exception_test exception.cpp)
target_link_libraries(exception_test PRIVATE fmt)
set_common_options(exception_test)
enable_sanitizers(exception_test)
install(TARGETS exception_test DESTINATION bin)

# -------------------------------
# SIMD copy executable for ARM processors
if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    add_executable(simd_copy simd_copy.cpp)
    target_link_libraries(simd_copy)
    set_common_options(simd_copy)
    enable_sanitizers(simd_copy)
    install(TARGETS simd_copy DESTINATION bin)
endif()

# -------------------------------
# Double buffer swapper test
add_executable(double_buffer_swapper_test double_buffer_swapper_test.cpp)
target_link_libraries(double_buffer_swapper_test PRIVATE fmt)
set_common_options(double_buffer_swapper_test)
enable_sanitizers(double_buffer_swapper_test)
install(TARGETS double_buffer_swapper_test DESTINATION bin)

# -------------------------------
# Async runner library and test
add_library(async_runner async_runner.cpp)
target_link_libraries(async_runner PRIVATE fmt)
install(TARGETS async_runner DESTINATION lib)

add_executable(async_runner_test async_runner_test.cpp)
target_link_libraries(async_runner_test PRIVATE fmt async_runner)
set_common_options(async_runner_test)
enable_sanitizers(async_runner_test)
install(TARGETS async_runner_test DESTINATION bin)
