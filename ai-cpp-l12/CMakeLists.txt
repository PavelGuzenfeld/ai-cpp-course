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
    # No -warnings-as-errors=* here: it overrides .clang-tidy's WarningsAsErrors
    # and promotes a naming nit in polynomial_flags.cpp to a build error. The
    # config file is the policy -- that is the thing the lesson is teaching.
    set(CMAKE_CXX_CLANG_TIDY
        ${CLANG_TIDY_EXE}
        --extra-arg=-std=c++23)
    message(STATUS "clang-tidy: ${CLANG_TIDY_EXE}")
  else()
    # A warning here is how this lesson went unchecked: the toggle was on and
    # the exercise silently had no teeth. Asked for explicitly, it must exist.
    message(FATAL_ERROR "ENABLE_CLANG_TIDY=ON but no clang-tidy-19 or clang-tidy on PATH")
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
