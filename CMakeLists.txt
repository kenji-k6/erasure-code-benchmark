cmake_minimum_required(VERSION 3.16)


# Prevent in-source builds
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "In-source builds are not allowed. Please run cmake from a 'build' directory.")
endif()

project(ec-benchmark LANGUAGES CXX CUDA)

# Set C++ standard to 20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Set architecture to sm_70 (since we have Tesla V100s)
set(CMAKE_CUDA_ARCHITECTURES 70)

# Set proper compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -O3 -march=native")

find_package(OpenMP REQUIRED) # needed for leopard

# Add source files
set (SOURCES
  src/main.cpp
  src/benchmark/benchmark_functions.cpp
  src/benchmark/benchmark_reporters.cpp
  src/benchmark/benchmark_utils.cpp

  src/utils/utils.cpp
  src/utils/cuda_utils.cu

  src/baseline/xorec.cpp

  src/algorithms/xorec_benchmark.cpp
  src/algorithms/cm256_benchmark.cpp
  src/algorithms/isal_benchmark.cpp
  src/algorithms/leopard_benchmark.cpp
  src/algorithms/wirehair_benchmark.cpp
)


# Create the ec-benchmark executable
add_executable(ec-benchmark ${SOURCES})



# Maybe in the future: link necessary libraries / check if submodules are initialized
set_target_properties(ec-benchmark PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)

# Google benchmarking library
set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)
set(CMAKE_BUILD_TYPE Release)
add_subdirectory(libraries/google_benchmark)

# leopard
add_subdirectory(libraries/leopard)

# cm256
add_subdirectory(libraries/cm256)

# wirehair (have to set BUILD_TESTS to OFF)
set(BUILD_TESTS OFF)
message(STATUS "BUILD_TESTS in wirehair: ${BUILD_TESTS}")
add_subdirectory(libraries/wirehair)


add_library(isal STATIC IMPORTED)
set_target_properties(isal PROPERTIES
  IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libraries/isa-l/.libs/libisal.a
)



# Include directories
target_include_directories(ec-benchmark PRIVATE
  ${CMAKE_SOURCE_DIR}/src/algorithms
  ${CMAKE_SOURCE_DIR}/src/baseline
  ${CMAKE_SOURCE_DIR}/src/utils
  ${CMAKE_SOURCE_DIR}/src/benchmark
  ${CMAKE_SOURCE_DIR}/libraries/leopard
  ${CMAKE_SOURCE_DIR}/libraries/isa-l/include
)


# CUDA Stuff
set(CUDA_TOOLKIT_ROOT_DIR /apps/ault/spack/opt/spack/linux-centos8-zen2/gcc-10.2.0/cuda-11.4.0-udyaakpt7oztg7gnj764dhkhdf5ory5d)
find_package(CUDA REQUIRED)

# Include CUDA headers and libraries
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

# Linking the Libraries
target_link_libraries(ec-benchmark PRIVATE
  libleopard
  cm256
  wirehair
  isal
  benchmark::benchmark
  OpenMP::OpenMP_CXX
  ${CUDA_LIBRARIES}
)

