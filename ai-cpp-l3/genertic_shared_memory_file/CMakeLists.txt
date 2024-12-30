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


# Function to apply common compile and link options
function(set_common_options target)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${target} PRIVATE
            -O3                # Max optimization
            -g3                # Debugging information
            -march=native      # Enable all CPU-specific optimizations
            -funroll-loops     # Unroll loops for speed
            -fopenmp           # Enable OpenMP for multi-threading (if applicable)
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

# Add the Python module using nanobind
add_library(Share_memory_image_producer_consumer MODULE
    flat_shm_img_producer_consumer_py.cpp # Source file for Python bindings
    flat_shm_producer_consumer.hpp
    image.hpp
)

# Link dependencies
target_link_libraries(Share_memory_image_producer_consumer PRIVATE pybind11::module fmt)

# Apply common options to the module
set_common_options(Share_memory_image_producer_consumer)

# Set properties for the Python module
set_target_properties(Share_memory_image_producer_consumer PROPERTIES PREFIX "" SUFFIX ".so")

install(TARGETS Share_memory_image_producer_consumer DESTINATION lib)

# Install the Python module to site-packages
install(TARGETS Share_memory_image_producer_consumer
    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages
)

# Add the test executable
add_executable(flat_shm_test flat_shm_test.cpp)

# Link dependencies for the test executable
target_link_libraries(flat_shm_test PRIVATE fmt)

# Apply common options to the test executable
set_common_options(flat_shm_test)

# Install the test executable
install(TARGETS flat_shm_test DESTINATION bin)
