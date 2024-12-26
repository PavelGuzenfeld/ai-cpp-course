cmake_minimum_required(VERSION 3.15)
project(flat_shared_memory)

# Require C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


find_package(fmt REQUIRED)
# Find Python and Pybind11
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

# # Add the library
# add_library(${PROJECT_NAME} MODULE ${PROJECT_NAME}.cpp)

# # Link libraries
# target_link_libraries(${PROJECT_NAME} PRIVATE pybind11::module)

# # Set properties for Python module
# set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".so")

# # Install python library
# install(TARGETS ${PROJECT_NAME}
#     DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages
# )

add_executable(flat_shm_test flat_shm_test.cpp)
set_common_options(flat_shm_test)
target_link_libraries(flat_shm_test PRIVATE fmt)
install(TARGETS flat_shm_test DESTINATION bin)