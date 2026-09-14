cmake_minimum_required(VERSION 3.20)
project(flat-type)

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

# Find dependencies
find_package(fmt REQUIRED)

# Create an INTERFACE library target for the header-only library.
add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME}
    INTERFACE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
)

# Install header files
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION include
)

# Export the target so that consumers can use the namespaced alias.
install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    INCLUDES DESTINATION include
)

install(
    EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION lib/cmake/${PROJECT_NAME}
)

# Configure the package configuration file.
include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  DESTINATION lib/cmake/${PROJECT_NAME}
)

# Test executable
add_executable(${PROJECT_NAME}_test test/flat_type_test.cpp)
target_include_directories(${PROJECT_NAME}_test PRIVATE include)
target_link_libraries(${PROJECT_NAME}_test PRIVATE fmt)

set_debug_options(${PROJECT_NAME}_test)
enable_sanitizers(${PROJECT_NAME}_test)

# Export the target
install(
    TARGETS ${PROJECT_NAME}_test
    EXPORT ${PROJECT_NAME}Targets
    DESTINATION bin
)