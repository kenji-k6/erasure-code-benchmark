cmake_minimum_required(VERSION 3.16)
project(ecc-benchmark)

# Prevent in-source builds
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "In-source builds are not allowed. Please run cmake from a 'build' directory.")
endif()

# Set C++ standard to 23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Leopard todo...

# Add source files
set (SOURCES
  benchmark/main.cpp
  benchmark/benchmark_runner.cpp
  benchmark/error_injector.cpp
  benchmark/utils.cpp
)



# Create the ebnchmark executable
add_executable(ecc-benchmark ${SOURCES})



#Maybe in the future: link necessary libraries / check if submodules are initialized
set_target_properties(ecc-benchmark PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)

