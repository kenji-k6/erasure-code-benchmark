#include "xorbaseline.h"
#include <cstring>
#include <iostream>


XORBaselineResult xor_encode(
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
    num_parity_blocks > XORBASELINE_MAX_PARITY_BLOCKS ||
    num_parity_blocks > num_data_blocks ||
    num_data_blocks % num_parity_blocks != 0
  ) {
        return XORBaseline_InvalidCounts;
  }

  if (
    reinterpret_cast<uintptr_t>((uintptr_t)data_buffer) % XORBASELINE_PTR_ALIGNMENT != 0 ||
    reinterpret_cast<uintptr_t>(parity_buffer) % XORBASELINE_PTR_ALIGNMENT != 0
  ) {
    return XORBaseline_InvalidAlignment;
  }

  // Zero out parity buffer
  std::memset(parity_buffer, 0, block_size * num_parity_blocks);

  #if defined(XORBASELINE_AVX512)
    // AVX-512 implementation
  #elif defined(XORBASELINE_AVX2)
    // AVX2 implementation
  #elif defined(XORBASELINE_AVX)
    // AVX implementation
  #else
    // Scalar implementation

    for (unsigned i = 0; i < num_parity_blocks; ++i) {
      uint64_t *parity_block = reinterpret_cast<uint64_t*>(parity_buffer + i * block_size);

      // j-th data block is part of xor of i-th parity block, if j % num_parity_blocks == i
      for (unsigned j = i; j < num_data_blocks; j += num_parity_blocks) {
        uint64_t *data_block = reinterpret_cast<uint64_t*>(data_buffer + + j * block_size);
        for (unsigned k = 0; k < block_size / sizeof(uint64_t); ++k) {
          parity_block[k] ^= data_block[k];
        }
      }
    }

  #endif
  return XORBaseline_Success;  
}


XORBaselineResult xor_decode(
  uint8_t *data_buffer,
  uint8_t *parity_buffer,
  uint32_t block_size,
  uint32_t num_data_blocks,
  uint32_t num_parity_blocks,
  std::bitset<256> block_bitmap
) {
  if (block_size < XORBASELINE_MIN_BLOCK_SIZE || block_size % XORBASELINE_BLOCK_SIZE_MULTIPLE != 0) {
    return XORBaseline_InvalidSize;
  }

  if (
    num_data_blocks + num_parity_blocks > XORBASELINE_MAX_TOTAL_BLOCKS ||
    num_data_blocks < XORBASELINE_MIN_DATA_BLOCKS ||
    num_data_blocks > XORBASELINE_MAX_DATA_BLOCKS ||
    num_parity_blocks < XORBASELINE_MIN_PARITY_BLOCKS ||
    num_parity_blocks > XORBASELINE_MAX_PARITY_BLOCKS ||
    num_parity_blocks > num_data_blocks ||
    num_data_blocks % num_parity_blocks != 0
  ) {
        return XORBaseline_InvalidCounts;
  }

  if (
    reinterpret_cast<uintptr_t>((uintptr_t)data_buffer) % XORBASELINE_PTR_ALIGNMENT != 0 ||
    reinterpret_cast<uintptr_t>(parity_buffer) % XORBASELINE_PTR_ALIGNMENT != 0
  ) {
    return XORBaseline_InvalidAlignment;
  }

  if ((block_bitmap & compare_bitmap).count() == num_data_blocks) return XORBaseline_Success; // No need to decode, all data blocks are present

  // Check recoverability
  // if more than one block per "set" is lost, we can't recover => immediately return
  std::bitset<128> block_lost_bitmap;
  for (unsigned i = 0; i < num_parity_blocks; ++i) {
    if (!block_bitmap.test(128 + i)) block_lost_bitmap.set(i);
  }
  
  
  for (unsigned j = 0; j < num_data_blocks; ++j) {
    uint32_t parity_idx = j % num_parity_blocks;
    if (!block_bitmap.test(j)) {
      if (block_lost_bitmap.test(parity_idx)) return XORBaseline_DecodeFailure;
      block_lost_bitmap.set(parity_idx);
    }
  }

  
  

  // Recovery possible => Decode



  #if defined(XORBASELINE_AVX512)
    // AVX-512 implementation
  #elif defined(XORBASELINE_AVX2)
    // AVX2 implementation
  #elif defined(XORBASELINE_AVX)
    // AVX implementation
  #else
    // Scalar implementation
    // Decode the data if possible
    for (unsigned i = 0; i < num_data_blocks; ++i) {
      if (block_bitmap[i]) continue; // Data block is present

      // Block is lost => copy parity block to data block and xor with rest of blocks
      uint64_t *recover_block = reinterpret_cast<uint64_t*>(data_buffer + i * block_size);
      uint8_t *parity_block = parity_buffer + (i % num_parity_blocks) * block_size;

      std::memcpy(recover_block, parity_block, block_size);

      for (unsigned j = i%num_parity_blocks; j < num_data_blocks; j+= num_parity_blocks) {
        if (i == j) continue;

        uint64_t *data_block = reinterpret_cast<uint64_t*>(data_buffer + j * block_size);
        for (unsigned k = 0; k < block_size / sizeof(uint64_t); ++k) {
          recover_block[k] ^= data_block[k];
        }
      }
    }
  #endif

  return XORBaseline_Success;
}