/**
 * @file utils.hpp
 * @brief Utility functions for benchmarking, random number generation,
 * data corruption detection and memory allocation (amongst others).
 */

#ifndef UTILS_HPP
#define UTILS_HPP

/// @attention THIS MACRO TOGGLES WHETHER TO CORRUPTION CHECK AND VALIDATION OF DATA IS MADE OR NOT
/// Since validation incurs a significant overhead, one can get results faster by disabling it.
#define ENABLE_VALIDATION 0


#include <algorithm>
#include <cstdint>
#include <immintrin.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>


//  Constants for fixed values
constexpr size_t ALIGNMENT = 64;
constexpr size_t RANDOM_SEED = 1896;
constexpr size_t MIN_DATA_BLOCK_SIZE = 2;


namespace ECLimits {
  constexpr size_t XOREC_BLOCK_ALIGNMENT = 64;

  constexpr size_t CM256_MAX_TOT_BLOCKS = 256;

  constexpr size_t ISAL_MIN_BLOCK_SIZE = 64;
  constexpr size_t ISAL_MAX_DATA_BLOCKS = 256;
  constexpr size_t ISAL_MAX_TOT_BLOCKS = 256;

  constexpr size_t LEOPARD_MAX_TOT_BLOCKS = 65'536;
  constexpr size_t LEOPARD_BLOCK_ALIGNMENT = 64;

  constexpr size_t WIREHAIR_MIN_DATA_BLOCKS = 2;
  constexpr size_t WIREHAIR_MAX_DATA_BLOCKS = 64'000;
}


/**
 * @class PCGRandom
 * @brief Implements the PCG random number generator for benchmarking.
 * @see https://www.pcg-random.org/
 */
class PCGRandom {
private:
  uint64_t state; // Internal RNG state
  uint64_t inc;   // Increment value (must be odd)
public:
  PCGRandom(uint64_t seed, uint64_t seq);
  uint32_t next();  // Generate a random 32-bit number
};


/**
 * @brief Writes a data pattern to a block for corruption detection.
 * @param block_ptr Pointer to the block memory.
 * @param size Size of the block in bytes.
 * @return 0 on success, nonzero on failure.
 * 
 * Uses a timestamp as a random seed to generate a unique pattern.
 */
int write_validation_pattern(uint8_t* block_ptr, size_t bytes);


/**
 * @brief Check's if a block's content has been corrupted.
 * @param block_ptr Pointer to the block memory.
 * @param size Size of the block in bytes.
 * @return True if the block is not corrupted, false otherwise.
 */
bool validate_block(const uint8_t* block_ptr, size_t bytes);


/**
 * @brief Selects k unique indices from range [0, num_data_blocks+num_parityblocks)
 * to simulate loss of blocks.
 * @attention To properly benchmark CM256 and XOR-EC, the lost blocks returned always make a recoverable set.
 * @param num_data_blocks Number of data blocks.
 * @param num_parity_blocks Number of parity blocks.
 * @param num_lost_blocks Number of blocks to select.
 * @param block_bitmap Pointer to the bitmap to store lost blocks
 */
void select_lost_blocks(size_t num_data_blocks, size_t num_parity_blocks, size_t num_lost_blocks, uint8_t* block_bitmap);


/**
 * @brief Helper function to throw an error message.
 * @param message Error message
 */
[[noreturn]] void throw_error(const std::string& message);


/**
 * @brief Helper function to convert a string to lowercase.
 * @param str Input string
 * @return lowercase version
 */
std::string to_lower(std::string str);


/// @attention Functions below are utility function for safe memory allocation
template <typename T>
using DeleterFunc = void(*)(T*);

template <typename T>
void mm_deleter(T* ptr) {
  if (ptr) _mm_free(ptr);
}

template <typename T>
void cuda_deleter(T* ptr) {
  if (ptr) {
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) throw_error("Failed to free CUDA memory");
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) throw_error("Failed to synchronize CUDA device");
  }
}

template <typename T>
void cuda_host_deleter(T* ptr) {
  if (ptr) {
    cudaError_t err = cudaFreeHost(ptr);
    if (err != cudaSuccess) throw_error("Failed to free CUDA memory");
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) throw_error("Failed to synchronize CUDA device");
  }
}

template <typename T>
std::unique_ptr<T[], DeleterFunc<T>> make_unique_aligned(size_t count = 1) {
  static_assert(std::is_trivially_destructible_v<T>, "Type must be trivially destructible");
  void* mem = _mm_malloc(count * sizeof(T), ALIGNMENT);
  if (!mem) throw std::bad_alloc();
  return std::unique_ptr<T[], DeleterFunc<T>>(static_cast<T*>(mem), mm_deleter);
}

template <typename T>
std::unique_ptr<T[], DeleterFunc<T>> make_unique_cuda(size_t count = 1) {
  static_assert(std::is_trivially_destructible_v<T>, "Type must be trivially destructible");
  void* mem = nullptr;
  cudaError_t err = cudaMalloc(&mem, count * sizeof(T));
  if (err != cudaSuccess) throw_error("Failed to allocate CUDA memory");
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) throw_error("Failed to synchronize CUDA device");
  return std::unique_ptr<T[], DeleterFunc<T>>(static_cast<T*>(mem), cuda_deleter);
}

template <typename T>
std::unique_ptr<T[], DeleterFunc<T>> make_unique_cuda_host(size_t count = 1) {
  static_assert(std::is_trivially_destructible_v<T>, "Type must be trivially destructible");
  void* mem = nullptr;
  cudaError_t err = cudaMallocHost(&mem, count * sizeof(T));
  if (err != cudaSuccess) throw_error("Failed to allocate CUDA memory");
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) throw_error("Failed to synchronize CUDA device");
  return std::unique_ptr<T[], DeleterFunc<T>>(static_cast<T*>(mem), cuda_host_deleter);
}

template <typename T>
std::unique_ptr<T[], DeleterFunc<T>> make_unique_cuda_managed(size_t count = 1, unsigned int flags = 1U) {
  static_assert(std::is_trivially_destructible_v<T>, "Type must be trivially destructible");
  void* mem = nullptr;
  cudaError_t err = cudaMallocManaged(&mem, count * sizeof(T), flags);
  if (err != cudaSuccess) throw_error("Failed to allocate CUDA memory");
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) throw_error("Failed to synchronize CUDA device");
  return std::unique_ptr<T[], DeleterFunc<T>>(static_cast<T*>(mem), cuda_deleter);
}

#endif // UTILS_HPP