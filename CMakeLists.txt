cmake_minimum_required(VERSION 3.16)
project(ecc-benchmark)
cmake_policy(SET CMP0079 NEW)
# Prevent in-source builds
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "In-source builds are not allowed. Please run cmake from a 'build' directory.")
endif()

# Set C++ standard to 23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Include directories for ECC libraries
include_directories(
  # libraries/aff3ct/include
  # libraries/aff3ct/lib/cli/src
  # libraries/aff3ct/lib/streampu/include
  # libraries/aff3ct/lib/streampu/lib/rang/include
  # libraries/aff3ct/lib/MIPP/src
  libraries/cm256/include
  libraries/isa-l
  libraries/leopard
  libraries/wirehair/include
)

# add_subdirectory(libraries/aff3ct)

# Include the ECC libraries
add_library(leopard STATIC 
    libraries/leopard/leopard.cpp
    libraries/leopard/LeopardCommon.cpp
    libraries/leopard/LeopardFF8.cpp
    libraries/leopard/LeopardFF16.cpp
)

add_library(cm256 STATIC
    libraries/cm256/src/cm256.cpp
    libraries/cm256/src/gf256.cpp
)

add_library(wirehair STATIC
    libraries/wirehair/wirehair.cpp
    libraries/wirehair/WirehairCodec.cpp
    libraries/wirehair/gf256.cpp
    libraries/wirehair/WirehairTools.cpp
)

# add_library(aff3ct STATIC
#     libraries/aff3ct/src/main.cpp
# )

# Add source files
set (SOURCES
  benchmark/main.cpp
  benchmark/benchmark_runner.cpp
  benchmark/error_injector.cpp
  benchmark/utils.cpp
)



# Create the ebnchmark executable
add_executable(ecc-benchmark ${SOURCES})


set(CMAKE_THREAD_PREFER_PTHREAD ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Link with affect
set(AFF3CT_DIR "${CMAKE_SOURCE_DIR}/build/cmake/Modules/")
find_package(AFF3CT REQUIRED)
#cp ../../lib/aff3ct/build/lib/cmake/aff3ct-*/*
#Maybe in the future: link necessary libraries / check if submodules are initialized
set_target_properties(ecc-benchmark PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)
target_link_libraries(ecc-benchmark PRIVATE fmt::fmt)
# target_link_libraries(ecc-benchmark PRIVATE aff3ct)
target_link_libraries(ecc-benchmark PRIVATE cm256)
# target_link_libraries(ecc-benchmark PRIVATE isa-l)
target_link_libraries(ecc-benchmark PRIVATE leopard)
target_link_libraries(ecc-benchmark PRIVATE wirehair)
target_link_libraries(ecc-benchmark PRIVATE aff3ct::aff3ct-static-lib)


# needed for Gnuradio
find_package(fmt REQUIRED)

# needed for cm256/aff3ct
find_package(OpenMP REQUIRED)
if(OpenMP_C_FOUND)
    target_link_libraries(ecc-benchmark PRIVATE OpenMP::OpenMP_C)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(ecc-benchmark PRIVATE OpenMP::OpenMP_CXX)
endif()