cmake_minimum_required(VERSION 3.16)
project(ecc-benchmark)

# Prevent in-source builds
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "In-source builds are not allowed. Please run cmake from a 'build' directory.")
endif()

# Set C++ standard to 23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED) # needed for leopard

# Add source files
set (SOURCES
  src/library_benchmark/baseline_benchmark.cpp
  src/utils.cpp
  src/main.cpp
  src/benchmark_reporters.cpp
  src/library_benchmark/leopard_benchmark.cpp
  src/library_benchmark/cm256_benchmark.cpp
  src/library_benchmark/wirehair_benchmark.cpp
  src/library_benchmark/isal_benchmark.cpp
)


# Create the ecc-benchmark executable
add_executable(ecc-benchmark ${SOURCES})



# Maybe in the future: link necessary libraries / check if submodules are initialized
set_target_properties(ecc-benchmark PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)

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
add_subdirectory(libraries/wirehair)


# aff3ct (have to set AFF3CT_COMPILE_EXE, AFF3CT_COMPILE_SHARED_LIB to OFF and AFF3CT_COMPILE_STATIC_LIB to ON)
# set(AFF3CT_COMPILE_EXE OFF)
# set(AFF3CT_COMPILE_SHARED_LIB OFF)
# set(AFF3CT_COMPILE_STATIC_LIB ON)
# add_subdirectory(libraries/aff3ct)


add_library(isal STATIC IMPORTED)
set_target_properties(isal PROPERTIES
  IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libraries/isa-l/.libs/libisal.a
)



# Include directories
target_include_directories(ecc-benchmark PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/libraries/leopard
  ${CMAKE_SOURCE_DIR}/libraries/isa-l/include
)

# Linking the Libraries
target_link_libraries(ecc-benchmark PRIVATE
  libleopard
  cm256
  wirehair
  # aff3ct-static-lib
  isal
  benchmark::benchmark
  OpenMP::OpenMP_CXX
)

