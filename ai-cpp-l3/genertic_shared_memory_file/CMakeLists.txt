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

# Fallback: Manually add the target if not found
if (NOT TARGET nanobind::nanobind)
    add_library(nanobind::nanobind INTERFACE IMPORTED)
    set_target_properties(nanobind::nanobind PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "/usr/local/nanobind/include"
    )
endif()

file(GLOB NANOBIND_SOURCES /usr/local/nanobind/src/*.cpp)



# Add an option to enable sanitizers
option(ENABLE_SANITIZERS "Enable Address/Undefined Behavior sanitizers" OFF)

# Function to apply sanitizer flags to a target
function(enable_sanitizers target)
    if(ENABLE_SANITIZERS)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_compile_options(${target} PRIVATE
                -fsanitize=address,undefined
            )
            target_link_options(${target} PRIVATE
                -fsanitize=address,undefined
            )
        elseif(MSVC)
            # Add equivalent MSVC sanitizer flags here if supported
        endif()
    endif()
endfunction()

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
            -Wstack-usage=4096 # Warn for stack usage exceeding 4KB
            -fstack-usage      # Generate stack usage reports
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

# Add the Python module
add_library(Share_memory_image_producer_consumer MODULE
    flat_shm_img_producer_consumer_py.cpp
    flat_shm_producer_consumer.hpp
    image.hpp
)

# Set target properties
set_target_properties(Share_memory_image_producer_consumer PROPERTIES PREFIX "" SUFFIX ".so")

# Link dependencies and enable sanitizers
target_link_libraries(Share_memory_image_producer_consumer PRIVATE pybind11::module fmt)
set_common_options(Share_memory_image_producer_consumer)
if (ENABLE_SANITIZERS)
    enable_sanitizers(Share_memory_image_producer_consumer)
endif()

# Install the Python module
install(TARGETS Share_memory_image_producer_consumer
    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages
)

# Add the test executable
add_executable(flat_shm_test flat_shm_test.cpp)
target_link_libraries(flat_shm_test PRIVATE fmt)
set_common_options(flat_shm_test)
enable_sanitizers(flat_shm_test)

# Install the test executable
install(TARGETS flat_shm_test DESTINATION bin)


add_executable(exception_test exception.cpp)
target_link_libraries(exception_test PRIVATE fmt)
set_common_options(exception_test)
enable_sanitizers(exception_test)

# Install the test executable
install(TARGETS exception_test DESTINATION bin)

# Your module
include_directories(/usr/local/nanobind/ext/robin_map/include)
set(NANOBIND_SRC /usr/local/nanobind/src/nb_combined.cpp)
add_library(my_module MODULE nanobind_example.cpp ${NANOBIND_SRC})
target_link_libraries(my_module PRIVATE nanobind::nanobind Python3::Python)

# Set target properties
set_target_properties(my_module PROPERTIES PREFIX "" SUFFIX ".so")

# Install the Python module
install(TARGETS my_module
    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages
)
