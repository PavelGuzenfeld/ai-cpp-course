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

# --- The -march value is an ISA name, so it does not travel between
# architectures. See README A.2: naming x86-64-v3 on aarch64 is not a
# no-op, it is a hard compile error, which is the lesson.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  set(POLY_ARCH_FLAG -march=x86-64-v3)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(POLY_ARCH_FLAG -march=armv8.2-a+simd)
else()
  set(POLY_ARCH_FLAG "")
  message(WARNING "l12: no -march baseline known for ${CMAKE_SYSTEM_PROCESSOR}; "
                  "poly_v3 and poly_fast build without one")
endif()

# --- Three separate polynomial binaries, one per flag set ---
# Same source, different compile options — lets you inspect the codegen diff
# and measure the runtime diff.

add_executable(poly_o2 polynomial_flags.cpp)
target_compile_options(poly_o2 PRIVATE -O2)

add_executable(poly_v3 polynomial_flags.cpp)
target_compile_options(poly_v3 PRIVATE -O3 ${POLY_ARCH_FLAG})

add_executable(poly_fast polynomial_flags.cpp)
target_compile_options(poly_fast PRIVATE -O3 -ffast-math ${POLY_ARCH_FLAG})

# --- Tidy exercise: build only the cleaned code.  Building dirty_code with
# clang-tidy enabled should fail — that is the point of the exercise.
# clean_code.cpp has no main() (it's a set of functions for clang-tidy to
# scan, not a program to run), so this must be an OBJECT library, not an
# executable -- add_executable fails to link on every default build.
add_library(clean_demo OBJECT clean_code.cpp)
