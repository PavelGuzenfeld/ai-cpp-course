cmake_minimum_required(VERSION 3.20)
project(nanobind-l18)

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

# --- Sanitizer options, mirroring ai-cpp-l11 / ai-cpp-l20 / ai-cpp-l15 ---
option(ENABLE_SANITIZERS "Enable ASAN and UBSAN on both modules" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer on both modules" OFF)

if(ENABLE_SANITIZERS)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address,undefined)
    message(STATUS "AddressSanitizer and UndefinedBehaviorSanitizer enabled")
endif()

if(ENABLE_TSAN)
    add_compile_options(-fsanitize=thread -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=thread)
    message(STATUS "ThreadSanitizer enabled")
endif()

# --- filter_native: box_blur + threshold_mask ---
nanobind_add_module(filter_native NB_STATIC filter_native.cpp)
set_target_properties(filter_native PROPERTIES PREFIX "" SUFFIX ".so")
install(TARGETS filter_native
    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)

# --- record_parser_native: bounds-checked wire-format parser ---
nanobind_add_module(record_parser_native NB_STATIC record_parser_native.cpp)
set_target_properties(record_parser_native PROPERTIES PREFIX "" SUFFIX ".so")
install(TARGETS record_parser_native
    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)

# --- fuzz_parser_libfuzzer: the actual coverage-guided fuzzer (clang only).
#     The deterministic sweep lives in test_validation.py, calling the same
#     record_parser.hpp logic through record_parser_native over a fixed
#     corpus — no separate standalone binary needed for that half. ---
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_executable(fuzz_parser_libfuzzer fuzz/fuzz_parser.cpp)
    set_target_properties(fuzz_parser_libfuzzer PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
    target_compile_options(fuzz_parser_libfuzzer PRIVATE
        -fsanitize=fuzzer,address,undefined -g -O1)
    target_link_options(fuzz_parser_libfuzzer PRIVATE
        -fsanitize=fuzzer,address,undefined)
    install(TARGETS fuzz_parser_libfuzzer DESTINATION bin)
else()
    message(STATUS "fuzz_parser_libfuzzer skipped: requires clang (-fsanitize=fuzzer)")
endif()
