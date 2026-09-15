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

# --- Standalone cache-warming demo ---
# Built as a plain executable (no pybind11); run it to see when the
# HFT "cache warming" pattern loses at the micro-benchmark scale.
add_executable(cache_warming_fails cache_warming_fails.cpp)
target_compile_options(cache_warming_fails PRIVATE -O2 -std=c++23)

# --- Stride view: Python-facing correct/sheared copy for unit testing ---
add_library(stride_view_native MODULE stride_view_native.cpp)
target_compile_features(stride_view_native PRIVATE cxx_std_23)
target_link_libraries(stride_view_native PRIVATE pybind11::module)
set_target_properties(stride_view_native PROPERTIES PREFIX "" SUFFIX ".so")
install(TARGETS stride_view_native
    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages
)

# --- Stride demo: standalone executable, deliberately overruns under
# --trigger-overrun. ENABLE_SANITIZERS builds it with ASan+UBSan so that
# overrun is caught rather than silently corrupting the heap -- kept as a
# plain executable (like cache_warming_fails) rather than a pybind11
# module, so sanitizing it never has to sanitize libpython too.
option(ENABLE_SANITIZERS "Build stride_demo with ASan+UBSan" OFF)
add_executable(stride_demo stride_demo.cpp)
target_compile_features(stride_demo PRIVATE cxx_std_23)
target_link_libraries(stride_demo PRIVATE ${OpenCV_LIBS})
if(ENABLE_SANITIZERS)
    target_compile_options(stride_demo PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer -g)
    target_link_options(stride_demo PRIVATE -fsanitize=address,undefined)
endif()
