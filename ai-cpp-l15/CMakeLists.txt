cmake_minimum_required(VERSION 3.20)
project(nanobind-l15)

# Require C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find dependencies
find_package(Python3 COMPONENTS Interpreter Development)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(nanobind CONFIG REQUIRED HINTS /usr/local/nanobind/cmake)
find_package(Threads REQUIRED)

# Silence warnings in nanobind headers by marking as SYSTEM include
include_directories(SYSTEM /usr/local/nanobind/include)

# --- Concurrency module (SPSC ring, GIL release demo) ---
nanobind_add_module(concurrency_native NB_STATIC concurrency_native.cpp)
target_link_libraries(concurrency_native PRIVATE Threads::Threads)
set_target_properties(concurrency_native PROPERTIES PREFIX "" SUFFIX ".so")
install(TARGETS concurrency_native
    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)

# --- tsan_race_demo: standalone executable, TSan only under -DENABLE_TSAN=ON
#     (mirrors ai-cpp-l11/asan_example and ai-cpp-l20/ubsan_example; TSan
#     cannot be combined with ASan/UBSan in the same binary) ---
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)

add_executable(tsan_race_demo tsan_race_demo.cpp)
target_link_libraries(tsan_race_demo PRIVATE Threads::Threads)
set_target_properties(tsan_race_demo PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
if(ENABLE_TSAN)
    target_compile_options(tsan_race_demo PRIVATE -fsanitize=thread -g -O1)
    target_link_options(tsan_race_demo PRIVATE -fsanitize=thread)
endif()
install(TARGETS tsan_race_demo DESTINATION bin)
