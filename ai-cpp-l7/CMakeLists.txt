cmake_minimum_required(VERSION 3.20)
project(ai_cpp_l7 LANGUAGES CXX)

# Require C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Python3
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# Find nanobind
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
    RESULT_VARIABLE nanobind_FOUND_RESULT
)

if(nanobind_FOUND_RESULT EQUAL 0)
    find_package(nanobind CONFIG REQUIRED PATHS ${nanobind_ROOT} NO_DEFAULT_PATH)
    set(HAVE_NANOBIND TRUE)
else()
    message(WARNING "nanobind not found — Python bindings will not be built. "
                    "Install with: pip install nanobind")
    set(HAVE_NANOBIND FALSE)
endif()

# Check for CUDA
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(HAVE_CUDA TRUE)
    message(STATUS "CUDA found — building GPU kernels")
    add_compile_definitions(HAVE_CUDA=1)
else()
    set(HAVE_CUDA FALSE)
    message(STATUS "CUDA not found — building CPU-only stubs")
    add_compile_definitions(HAVE_CUDA=0)
endif()

# ─── GPU Preprocess Module ───────────────────────────────────────────────────

if(HAVE_NANOBIND)
    if(HAVE_CUDA)
        # Build the CUDA fused preprocess kernel + bindings
        nanobind_add_module(gpu_preprocess
            gpu_preprocess.cu
        )
        target_compile_options(gpu_preprocess PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        )
        set_target_properties(gpu_preprocess PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "60;70;75;80;86;89;90"
        )
    else()
        # CPU-only stub with the same Python interface
        nanobind_add_module(gpu_preprocess
            gpu_preprocess_cpu.cpp
        )
        target_compile_options(gpu_preprocess PRIVATE -O3 -march=native)
    endif()
endif()

# ─── CPU Reference Preprocess (always built) ─────────────────────────────────

if(HAVE_NANOBIND)
    nanobind_add_module(gpu_preprocess_cpu_ref
        gpu_preprocess_cpu.cpp
    )
    target_compile_definitions(gpu_preprocess_cpu_ref PRIVATE MODULE_NAME=gpu_preprocess_cpu_ref)
    target_compile_options(gpu_preprocess_cpu_ref PRIVATE -O3 -march=native)
endif()

# ─── Pinned Memory Allocator ─────────────────────────────────────────────────

if(HAVE_NANOBIND)
    nanobind_add_module(pinned_allocator
        pinned_allocator.cpp
    )
    target_compile_options(pinned_allocator PRIVATE -O3 -march=native)
    if(HAVE_CUDA)
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(pinned_allocator PRIVATE CUDA::cudart)
    endif()
endif()

# ─── Installation ────────────────────────────────────────────────────────────

if(HAVE_NANOBIND)
    install(TARGETS gpu_preprocess_cpu_ref pinned_allocator
        LIBRARY DESTINATION "${Python3_SITEARCH}"
    )
    if(TARGET gpu_preprocess)
        install(TARGETS gpu_preprocess
            LIBRARY DESTINATION "${Python3_SITEARCH}"
        )
    endif()
endif()
