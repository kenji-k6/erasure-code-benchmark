#ifndef XOREC_UTILS_H
#define XOREC_UTILS_H
/**
 * @file xorec_utils.h
 * @brief Provides utility constants for both the CPU and the GPU-based XOR-EC implementations.
 */

#include <bitset>
#include <cstdint>

#define XOREC_RESTRICT __restrict

constexpr uint32_t XOREC_BLOCK_SIZE_MULTIPLE = 64;
constexpr uint32_t XOREC_MIN_BLOCK_SIZE = 64;

constexpr uint32_t XOREC_MIN_DATA_BLOCKS = 1;
constexpr uint32_t XOREC_MAX_DATA_BLOCKS = 128;
constexpr uint32_t XOREC_MIN_PARITY_BLOCKS = 1;
constexpr uint32_t XOREC_MAX_PARITY_BLOCKS = 128;
constexpr uint32_t XOREC_MAX_TOTAL_BLOCKS = 256;

/// Bitmap to check if all data blocks are available (no recovery needed)
const std::bitset<XOREC_MAX_TOTAL_BLOCKS> COMPLETE_DATA_BITMAP =(
  std::bitset<256>(0xFFFFFFFFFFFFFFFFULL)<<64) |
  std::bitset<256>(0xFFFFFFFFFFFFFFFFULL
  );

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
  Auto = 0,
  Scalar = 1,
  AVX = 2,
  AVX2 = 3
};



extern uint32_t DEVICE_ID;
extern uint32_t MAX_THREADS_PER_BLOCK;
extern uint32_t MAX_THREADS_PER_MULTIPROCESSOR;
extern uint32_t MAX_BLOCKS_PER_MULTIPROCESSOR;
extern uint32_t WARP_SIZE;
extern bool XOREC_GPU_INIT_CALLED;

void xorec_gpu_init();

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

static void inline and_bitmap(uint8_t * dst, const uint8_t *src1, const uint8_t *src2, uint32_t count) {
  uint64_t * dst_64 = reinterpret_cast<uint64_t*>(dst);
  const uint64_t * src1_64 = reinterpret_cast<const uint64_t*>(src1);
  const uint64_t * src2_64 = reinterpret_cast<const uint64_t*>(src2);
  uint32_t i;

  for (i = 0; i < count/4; ++i) {
    dst_64[i] = src1_64[i] & src2_64[i];
  }

  i *= 4;
  for (; i < count; ++i) {
    dst[i] = src1[i] & src2[i];
  }
}


#endif // XOREC_UTILS_H