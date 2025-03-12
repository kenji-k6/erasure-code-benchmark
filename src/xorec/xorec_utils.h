#ifndef XOREC_UTILS_H
#define XOREC_UTILS_H
/**
 * @file xorec_utils.h
 * @brief Provides utility constants for both the CPU and the GPU-based XOR-EC implementations.
 */

#include <bitset>
#include <cstdint>
#include <array>

#define XOREC_RESTRICT __restrict

constexpr uint32_t XOREC_BLOCK_SIZE_MULTIPLE = 64;
constexpr uint32_t XOREC_MIN_BLOCK_SIZE = 64;

constexpr uint32_t XOREC_MIN_DATA_BLOCKS = 1;
constexpr uint32_t XOREC_MAX_DATA_BLOCKS = 128;
constexpr uint32_t XOREC_MIN_PARITY_BLOCKS = 1;
constexpr uint32_t XOREC_MAX_PARITY_BLOCKS = 128;
constexpr uint32_t XOREC_MAX_TOTAL_BLOCKS = 256;

/// Bitmap to check if all data blocks are available (no recovery needed)
std::array<uint8_t, XOREC_MAX_TOTAL_BLOCKS> COMPLETE_DATA_BITMAP = { 0 };


  /**
 * @enum XORResult
 * @brief Represents the result status of encoding and decoding operations.
 */
enum class XorecResult {
  Success = 0,
  InvalidSize = 1,
  InvalidCounts = 2,
  InvalidAlignment = 3,
  DecodeFailure = 4,
  KernelFailure = 5
};

/**
 * @enum XORVersion
 * @brief Allows to specify which version of the implementations to use
 */
enum class XorecVersion {
  Scalar = 0,
  AVX = 1,
  AVX2 = 2
};


static XorecResult inline xorec_check_args(uint32_t block_size, uint32_t num_data_blocks, uint32_t num_parity_blocks) {
  if (block_size < XOREC_MIN_BLOCK_SIZE || block_size % XOREC_BLOCK_SIZE_MULTIPLE != 0) {
    return XorecResult::InvalidSize;
  }

  if (
    num_data_blocks + num_parity_blocks > XOREC_MAX_TOTAL_BLOCKS ||
    num_data_blocks < XOREC_MIN_DATA_BLOCKS ||
    num_data_blocks > XOREC_MAX_DATA_BLOCKS ||
    num_parity_blocks < XOREC_MIN_PARITY_BLOCKS ||
    num_parity_blocks > XOREC_MAX_PARITY_BLOCKS ||
    num_parity_blocks > num_data_blocks ||
    num_data_blocks % num_parity_blocks != 0
  ) {
        return XorecResult::InvalidCounts;
  }
  return XorecResult::Success;
}

static void inline and_bitmap(uint8_t * dst, const uint8_t *src1, const uint8_t *src2, uint32_t len) {
  uint64_t * dst_64 = reinterpret_cast<uint64_t*>(dst);
  const uint64_t * src1_64 = reinterpret_cast<const uint64_t*>(src1);
  const uint64_t * src2_64 = reinterpret_cast<const uint64_t*>(src2);
  uint32_t i;

  for (i = 0; i < len/sizeof(uint64_t); ++i) {
    dst_64[i] = src1_64[i] & src2_64[i];
  }

  i *= sizeof(uint64_t);
  for (; i < len; ++i) {
    dst[i] = src1[i] & src2[i];
  }
}

static int inline bit_count(const uint8_t * bitmap, uint32_t len) {
  int count = 0;
  const uint32_t * bitmap_32 = reinterpret_cast<const uint32_t*>(bitmap);

  uint32_t i;

  for (i = 0; i < len/sizeof(uint32_t); ++i) {
    count += __builtin_popcount(bitmap_32[i]);
  }

  i *= sizeof(uint32_t);
  for (; i < len; ++i) {
    count += __builtin_popcount(bitmap[i]);
  }
  return count;
}
#endif // XOREC_UTILS_H