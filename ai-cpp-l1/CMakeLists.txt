cmake_minimum_required(VERSION 3.20)
project(portable_simd_sum_vectors)

# Require C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Python and Pybind11
find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Find TBB for parallel execution
find_package(TBB REQUIRED)

# Add the library
add_library(${PROJECT_NAME} MODULE ${PROJECT_NAME}.cpp)

# Set compilation options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE
        -O3                # Max optimization
        -march=native      # Enable all CPU-specific optimizations
        -funroll-loops     # Unroll loops for speed
        -fopenmp           # Enable OpenMP for multi-threading (if applicable)
        # -flto              # Link-time optimization
        -ffast-math        # Allow unsafe but faster math optimizations
    )
    target_link_options(${PROJECT_NAME} PRIVATE
        -flto              # Enable link-time optimization
    )
elseif(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        /O2                # Max optimization
        /GL                # Whole program optimization
    )
    target_link_options(${PROJECT_NAME} PRIVATE
        /LTCG              # Link-time code generation
    )
endif()

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE pybind11::module TBB::tbb)

# Set properties for Python module
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".so")

# Install the library
install(TARGETS ${PROJECT_NAME}
    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages
)
