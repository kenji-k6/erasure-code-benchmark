cmake_minimum_required(VERSION 3.16)


# Prevent in-source builds to keep build artifacts separate
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
message(FATAL_ERROR "In-source builds are not allowed. Please run cmake from a 'build' directory.")
endif()

# If not specified otherwise, set the build type to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Set the project name and languages
project(ec-benchmark LANGUAGES CXX CUDA)

# Set C++ standard to 20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard and compilation settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)


# Set architecture to sm_70 (since we have Tesla V100s)
set(CMAKE_CUDA_ARCHITECTURES 70)

# Set proper compiler flags (aggresive optimizations & SIMD)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -O3 -march=native -funroll-loops -mavx -mavx2 -mavx512f")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")

# Required for Leopard
find_package(OpenMP REQUIRED)

# Add source files
set (SOURCES
  src/algorithms/abstract_bm.cpp
  src/algorithms/cm256_bm.cpp
  src/algorithms/isal_bm.cpp
  src/algorithms/leopard_bm.cpp
  src/algorithms/xorec_bm.cpp
  src/algorithms/xorec_unified_ptr_bm.cpp
  src/algorithms/xorec_gpu_ptr_bm.cpp
  src/algorithms/xorec_gpu_cmp_bm.cpp

  src/benchmark/bm_config.cpp
  src/benchmark/runners.cpp
  
  src/reporters/console_reporter.cpp
  src/reporters/csv_reporter.cpp
  
  src/utils/benchmark_cli.cpp
  src/utils/benchmark_suite.cpp
  src/utils/utils.cpp

  src/xorec/xorec_gpu_cmp.cu
  src/xorec/xorec_utils.cpp
  src/xorec/xorec.cpp

  src/main.cpp
)


# Create the ec-benchmark executable
add_executable(ec-benchmark ${SOURCES})

set_target_properties(ec-benchmark PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build
)

# Google benchmarking library (download dependencies, disable tests)
# Enable libpfm for benchmarking (needed for counting cycles, instructions, etc.)
set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON CACHE BOOL "Enable downloading of dependencies" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable testing in Google Benchmark" FORCE)
add_subdirectory(libraries/google_benchmark)

# leopard
add_subdirectory(libraries/leopard)

# cm256
add_subdirectory(libraries/cm256)


add_library(isal STATIC IMPORTED)
set_target_properties(isal PROPERTIES
  IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libraries/isa-l/bin/isa-l.a
)


# Include directories
target_include_directories(ec-benchmark PRIVATE
  ${CMAKE_SOURCE_DIR}/src/algorithms
  ${CMAKE_SOURCE_DIR}/src/xorec
  ${CMAKE_SOURCE_DIR}/src/utils
  ${CMAKE_SOURCE_DIR}/src/benchmark
  ${CMAKE_SOURCE_DIR}/src/reporters
  ${CMAKE_SOURCE_DIR}/libraries/leopard
  ${CMAKE_SOURCE_DIR}/libraries/isa-l/include
  ${CMAKE_SOURCE_DIR}/libraries/perfmon2-libpfm4/include
)


# Ensure CUDA is available
find_package(CUDA REQUIRED)

# Include CUDA headers and libraries
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

# Linking the Libraries
target_link_libraries(ec-benchmark PRIVATE
  libleopard
  cm256
  isal
  benchmark::benchmark
  OpenMP::OpenMP_CXX
  ${CUDA_LIBRARIES}
)

