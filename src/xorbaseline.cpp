#include "xorbaseline.h"


XORBaselineResult encode(
  uint8_t *data_buffer,
  uint8_t *parity_buffer,
  uint32_t block_size,
  uint32_t num_data_blocks,
  uint32_t num_parity_blocks
) {
  if (block_size < XORBASELINE_MIN_BLOCK_SIZE || block_size % XORBASELINE_BLOCK_SIZE_MULTIPLE != 0) {
    return XORBaseline_InvalidSize;
  }

  if (
    num_data_blocks + num_parity_blocks > XORBASELINE_MAX_TOTAL_BLOCKS ||
    num_data_blocks < XORBASELINE_MIN_DATA_BLOCKS ||
    num_data_blocks > XORBASELINE_MAX_DATA_BLOCKS ||
    num_parity_blocks < XORBASELINE_MIN_PARITY_BLOCKS ||
    num_parity_blocks > XORBASELINE_MAX_PARITY_BLOCKS
  ) {
        return XORBaseline_InvalidCounts;
  }

  if (
    reinterpret_cast<uintptr_t>((uintptr_t)data_buffer) % XORBASELINE_PTR_ALIGNMENT != 0 ||
    reinterpret_cast<uintptr_t>(parity_buffer) % XORBASELINE_PTR_ALIGNMENT != 0
  ) {
    return XORBaseline_InvalidAlignment;
  }

  #if defined(XORBASELINE_AVX512)
    // AVX-512 implementation
  #elif defined(XORBASELINE_AVX2)
    // AVX2 implementation
  #elif defined(XORBASELINE_AVX)
    // AVX implementation
  #else
    // Scalar implementation

    for (unsigned i = 0; i < num_parity_blocks; i++) {
    
    }

  #endif


  return XORBaseline_Success;  
}