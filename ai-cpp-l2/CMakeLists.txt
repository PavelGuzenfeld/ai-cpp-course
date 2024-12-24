cmake_minimum_required(VERSION 3.15)
project(cpp_image_processor)

# Find Python and Pybind11
find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(xsimd REQUIRED)



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

# include opencv headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE pybind11::module TBB::tbb ${OpenCV_LIBS})

# Set properties for Python module
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".so")

# Install the library
install(TARGETS ${PROJECT_NAME}
    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages
)
