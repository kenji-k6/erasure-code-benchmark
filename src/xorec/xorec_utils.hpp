#ifndef XOREC_UTILS_HPP
#define XOREC_UTILS_HPP
/**
 * @file xorec_utils.h
 * @brief Provides utility constants for both the CPU and the GPU-based XOR-EC implementations.
 */

#include <string>
#include <vector>


#define XOREC_RESTRICT __restrict
extern std::vector<uint8_t> COMPLETE_DATA_BITMAP;

/// @brief Constants for the XOR-EC algorithm(s)
constexpr size_t XOREC_BLOCK_SIZE_MULTIPLE = 256;
constexpr size_t XOREC_MIN_BLOCK_SIZE = 256;
constexpr size_t XOREC_MIN_DATA_BLOCKS = 1;
constexpr size_t XOREC_MIN_PARITY_BLOCKS = 1;
constexpr size_t XOREC_ALIGNMENT = 64;

  /**
 * @enum XORResult
 * @brief Represents the result status of encoding and decoding operations.
 */
enum class XorecResult {
  Success = 0,
  InvalidSize = 1,
  InvalidAlignment = 2,
  InvalidCounts = 3,
  DecodeFailure = 4
};

/**
 * @enum XORVersion
 * @brief Allows to specify which version of the implementations to use
 */
enum class XorecVersion {
  Scalar = 0,
  SSE2 = 1,
  AVX2 = 2,
  AVX512 = 3
};

/**
 * @brief Get the version name corresponding to the provided version
 * 
 * @param version 
 * @return std::string 
 */
std::string get_version_name(XorecVersion version);

/**
 * @brief Auxiliary function to check validity of the provided arguments
 * 
 * @param block_size Block size in bytes
 * @param num_data_blocks Number of data blocks
 * @param num_parity_blocks Number of parity blocks
 * @return XorecResult 
 */
static XorecResult inline xorec_check_args(
  const uint8_t* XOREC_RESTRICT data_buf,
  const uint8_t* XOREC_RESTRICT parity_buf,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks
) {

  uintptr_t data_addr = reinterpret_cast<uintptr_t>(data_buf);
  uintptr_t parity_addr = reinterpret_cast<uintptr_t>(parity_buf);
  if (data_addr % XOREC_ALIGNMENT != 0 || parity_addr % XOREC_ALIGNMENT != 0)
    return XorecResult::InvalidAlignment;

  if (block_size < XOREC_MIN_BLOCK_SIZE || block_size % XOREC_BLOCK_SIZE_MULTIPLE != 0)
    

  if (block_size < XOREC_MIN_BLOCK_SIZE || block_size % XOREC_BLOCK_SIZE_MULTIPLE != 0)
    return XorecResult::InvalidSize;

  if (num_data_blocks < XOREC_MIN_DATA_BLOCKS     ||
      num_parity_blocks < XOREC_MIN_PARITY_BLOCKS ||
      num_data_blocks % num_parity_blocks != 0)
    return XorecResult::InvalidCounts;

  return XorecResult::Success;
}


/**
 * @brief Auxiliary function to perform a bitwise AND operation between two bitmaps
 * 
 * @param dst Pointer to the destination bitmap
 * @param src1 Pointer to the first source bitmap
 * @param src2 Pointer to the second source bitmap
 * @param bytes Number of bytes to process
 */
static void inline and_bitmap(uint8_t* XOREC_RESTRICT dst, const uint8_t* XOREC_RESTRICT src1, const uint8_t* XOREC_RESTRICT src2, size_t bytes) {
  uint64_t * dst_64 = reinterpret_cast<uint64_t*>(dst);
  const uint64_t * src1_64 = reinterpret_cast<const uint64_t*>(src1);
  const uint64_t * src2_64 = reinterpret_cast<const uint64_t*>(src2);
  uint32_t i;

  for (i = 0; i < bytes/sizeof(uint64_t); ++i) {
    dst_64[i] = src1_64[i] & src2_64[i];
  }

  i *= sizeof(uint64_t);
  for (; i < bytes; ++i) {
    dst[i] = src1[i] & src2[i];
  }
}

/**
 * @brief Counts thenumber of set bits in a bitmap
 * 
 * @param bitmap Pointer to the bitmap
 * @param bytes Number of bytes to process
 * @return int 
 */
static size_t inline bit_count(const uint8_t* XOREC_RESTRICT bitmap, size_t bytes) {
  int count = 0;
  const uint32_t * bitmap_32 = reinterpret_cast<const uint32_t*>(bitmap);

  uint32_t i;

  for (i = 0; i < bytes/sizeof(uint32_t); ++i) {
    count += __builtin_popcount(bitmap_32[i]);
  }

  i *= sizeof(uint32_t);
  for (; i < bytes; ++i) {
    count += __builtin_popcount(bitmap[i]);
  }
  return count;
}

/**
 * @brief Auxiliary function to check if recovery is needed
 * 
 * @param block_bitmap Pointer to the block bitmap
 * @return true If recovery is needed
 * @return false else
 */
static bool inline require_recovery(size_t num_data_blocks, const uint8_t* XOREC_RESTRICT block_bitmap) {
  std::vector<uint8_t> temp(num_data_blocks, 0);
  and_bitmap(temp.data(), block_bitmap, COMPLETE_DATA_BITMAP.data(), num_data_blocks);

  return bit_count(temp.data(), num_data_blocks) != num_data_blocks;
}

/**
 * @brief Auxiliary function to check if the data is recoverable
 * 
 * @param block_bitmap Pointer to the block bitmap
 * @param num_data_blocks Number of data blocks
 * @param num_parity_blocks Number of parity blocks
 * @return true If the data is recoverable
 * @return false else
 */
static bool inline is_recoverable(size_t num_data_blocks, size_t num_parity_blocks, const uint8_t* XOREC_RESTRICT block_bitmap) {
  std::vector<uint8_t> parity_needed(num_data_blocks, 0);
  for (unsigned i = 0; i < num_parity_blocks; ++i) {
    if (!block_bitmap[num_data_blocks + i]) parity_needed[i] = 1;
  }

  for (unsigned j = 0; j < num_data_blocks; ++j) {
    uint32_t parity_idx = j % num_parity_blocks;

    if (!block_bitmap[j]) {
      if (parity_needed[parity_idx]) return false;
      parity_needed[parity_idx] = 1;
    }
  }
  return true;
}

#endif // XOREC_UTILS_HPP