cmake_minimum_required(VERSION 3.20)
project(nanobind-l20)

# Require C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find dependencies
find_package(Python3 COMPONENTS Interpreter Development)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(nanobind CONFIG REQUIRED HINTS /usr/local/nanobind/cmake)

# Silence warnings in nanobind headers by marking as SYSTEM include
include_directories(SYSTEM /usr/local/nanobind/include)

# --- Sanitizer option for debug builds ---
option(ENABLE_SANITIZERS "Enable ASAN and UBSAN" OFF)

if(ENABLE_SANITIZERS)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address,undefined)
    message(STATUS "AddressSanitizer and UndefinedBehaviorSanitizer enabled")
endif()

# --- Robustness module (NaN poisoning, log-sum-exp) ---
nanobind_add_module(robustness_native NB_STATIC robustness_native.cpp)
set_target_properties(robustness_native PROPERTIES PREFIX "" SUFFIX ".so")
install(TARGETS robustness_native
    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)

# --- ubsan_example: standalone executable, UB trapped only when
#     ENABLE_SANITIZERS is on (mirrors ai-cpp-l11/asan_example) ---
add_executable(ubsan_example ubsan_example.cpp)
set_target_properties(ubsan_example PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
if(ENABLE_SANITIZERS)
    # float-cast-overflow is not part of GCC's default -fsanitize=undefined
    # group and must be requested explicitly; -fno-sanitize-recover makes it
    # abort instead of printing and continuing.
    target_compile_options(ubsan_example PRIVATE
        -fsanitize=address,undefined,float-cast-overflow
        -fno-sanitize-recover=undefined,float-cast-overflow
        -fno-omit-frame-pointer -g -O0)
    target_link_options(ubsan_example PRIVATE
        -fsanitize=address,undefined,float-cast-overflow)
endif()
install(TARGETS ubsan_example DESTINATION bin)

# --- assert_example built twice: assertions active, and -DNDEBUG ---
add_executable(assert_example_checked assert_example.cpp)
set_target_properties(assert_example_checked PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)

add_executable(assert_example_ndebug assert_example.cpp)
set_target_properties(assert_example_ndebug PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
target_compile_definitions(assert_example_ndebug PRIVATE NDEBUG)

install(TARGETS assert_example_checked assert_example_ndebug DESTINATION bin)
