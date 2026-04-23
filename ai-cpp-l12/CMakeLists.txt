cmake_minimum_required(VERSION 3.20)
project(ai-cpp-l12)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- clang-tidy integration (optional, toggle with -DENABLE_CLANG_TIDY=ON) ---
option(ENABLE_CLANG_TIDY "Run clang-tidy on every compile" OFF)
if(ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY_EXE NAMES clang-tidy-19 clang-tidy)
  if(CLANG_TIDY_EXE)
    set(CMAKE_CXX_CLANG_TIDY
        ${CLANG_TIDY_EXE}
        -warnings-as-errors=*
        --extra-arg=-std=c++23)
    message(STATUS "clang-tidy: ${CLANG_TIDY_EXE}")
  else()
    message(WARNING "clang-tidy not found; continuing without it")
  endif()
endif()

# --- Three separate polynomial binaries, one per flag set ---
# Same source, different compile options — lets you inspect the codegen diff
# and measure the runtime diff.

add_executable(poly_o2 polynomial_flags.cpp)
target_compile_options(poly_o2 PRIVATE -O2)

add_executable(poly_v3 polynomial_flags.cpp)
target_compile_options(poly_v3 PRIVATE -O3 -march=x86-64-v3)

add_executable(poly_fast polynomial_flags.cpp)
target_compile_options(poly_fast PRIVATE -O3 -ffast-math -march=x86-64-v3)

# --- Tidy exercise: build only the cleaned code.  Building dirty_code with
# clang-tidy enabled should fail — that is the point of the exercise.
add_executable(clean_demo clean_code.cpp)
