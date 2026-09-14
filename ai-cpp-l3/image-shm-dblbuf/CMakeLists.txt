cmake_minimum_required(VERSION 3.20)
project(image-shm-dblbuf)

# Require C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build utilities
function(set_debug_options target)
    target_compile_options(${target} PRIVATE -g -O0)
    target_compile_definitions(${target} PRIVATE DEBUG)
endfunction()

function(enable_sanitizers target)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${target} PRIVATE -fsanitize=address,undefined,signed-integer-overflow,null,leak)
        target_link_options(${target} PRIVATE -fsanitize=address,undefined,signed-integer-overflow,null,leak)
    elseif(MSVC)
    endif()
endfunction()

function(set_release_options target)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${target} PRIVATE -O3)
        target_compile_definitions(${target} PRIVATE NDEBUG)
        target_link_options(${target} PRIVATE -flto)
    elseif(MSVC)
        target_compile_options(${target} PRIVATE /O2 /GL)
        target_link_options(${target} PRIVATE /LTCG)
    endif()
endfunction()

# Find dependencies
find_package(fmt REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(nanobind CONFIG REQUIRED HINTS /usr/local/nanobind/cmake)
find_package(single-task-runner REQUIRED)
find_package(double-buffer-swapper REQUIRED)
find_package(flat-type REQUIRED)
find_package(exception-rt REQUIRED)
find_package(shm REQUIRED)

# Silence warnings in nanobind headers by marking as SYSTEM include
include_directories(SYSTEM /usr/local/nanobind/include)

# Nanobind module
nanobind_add_module(image_shm_dblbuff
    NB_STATIC
    src/flat_shm_img_producer_consumer_nb.cpp
)

target_include_directories(image_shm_dblbuff PRIVATE include)
target_link_libraries(image_shm_dblbuff
    PRIVATE
        fmt
        single-task-runner::single-task-runner
        double-buffer-swapper::double-buffer-swapper
        flat-type::flat-type
        exception-rt::exception-rt
        shm::shm
)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set_debug_options(image_shm_dblbuff)
    enable_sanitizers(image_shm_dblbuff)
else()
    set_release_options(image_shm_dblbuff)
endif()

install(TARGETS image_shm_dblbuff
    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages
)

# Tests
include(FetchContent)
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.12
)
FetchContent_MakeAvailable(doctest)

add_executable(flat_shm_test test/flat_shm_test.cpp)
target_include_directories(flat_shm_test PRIVATE include)
target_link_libraries(flat_shm_test
    PRIVATE
        fmt
        doctest::doctest
        flat-type::flat-type
        exception-rt::exception-rt
        shm::shm
)
set_debug_options(flat_shm_test)
enable_sanitizers(flat_shm_test)
add_test(NAME flat_shm_test COMMAND flat_shm_test)

add_executable(shm_test test/shm_test.cpp)
target_include_directories(shm_test PRIVATE include)
target_link_libraries(shm_test
    PRIVATE
        fmt
        doctest::doctest
        flat-type::flat-type
        double-buffer-swapper::double-buffer-swapper
        single-task-runner::single-task-runner
        exception-rt::exception-rt
        shm::shm
)
set_debug_options(shm_test)
enable_sanitizers(shm_test)
add_test(NAME shm_test COMMAND shm_test)
