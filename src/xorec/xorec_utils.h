#ifndef XOREC_UTILS_H
#define XOREC_UTILS_H
/**
 * @file xorec_utils.h
 * @brief Provides utility constants for both the CPU and the GPU-based XOR-EC implementations.
 */

#include <bitset>
#include <cstdint>

#define XOR_RESTRICT __restrict

constexpr uint32_t XOR_BLOCK_SIZE_MULTIPLE = 64;
constexpr uint32_t XOR_MIN_BLOCK_SIZE = 64;

constexpr uint32_t XOR_MIN_DATA_BLOCKS = 1;
constexpr uint32_t XOR_MAX_DATA_BLOCKS = 128;
constexpr uint32_t XOR_MIN_PARITY_BLOCKS = 1;
constexpr uint32_t XOR_MAX_PARITY_BLOCKS = 128;
constexpr uint32_t XOR_MAX_TOTAL_BLOCKS = 256;

/// Bitmap to check if all data blocks are available (no recovery needed)
const std::bitset<XOR_MAX_TOTAL_BLOCKS> COMPLETE_DATA_BITMAP =(
  std::bitset<256>(0xFFFFFFFFFFFFFFFFULL)<<64) |
  std::bitset<256>(0xFFFFFFFFFFFFFFFFULL
  );

  /**
 * @enum XORResult
 * @brief Represents the result status of encoding and decoding operations.
 */
enum class XORResult {
  Success = 0,
  InvalidSize = 1,
  InvalidCounts = 2,
  InvalidAlignment = 3,
  DecodeFailure = 4,
  KernelFailure = 5
};

#endif // XOREC_UTILS_H