#include "xorbaseline.h"
#include <cstring>
#include <iostream>


/**
 * @file xorbaseline.cpp
 * @brief Implements XOR-based erasure encoding and decoding functions.
 */


XORResult xor_encode(
  const uint8_t *XOR_RESTRICT data_buffer,
  uint8_t *XOR_RESTRICT parity_buffer,
  uint32_t block_size,
  uint32_t num_data_blocks,
  uint32_t num_parity_blocks
) {
  if (block_size < XOR_MIN_BLOCK_SIZE || block_size % XOR_BLOCK_SIZE_MULTIPLE != 0) {
    return XORResult::InvalidSize;
  }

  if (
    num_data_blocks + num_parity_blocks > XOR_MAX_TOTAL_BLOCKS ||
    num_data_blocks < XOR_MIN_DATA_BLOCKS ||
    num_data_blocks > XOR_MAX_DATA_BLOCKS ||
    num_parity_blocks < XOR_MIN_PARITY_BLOCKS ||
    num_parity_blocks > XOR_MAX_PARITY_BLOCKS ||
    num_parity_blocks > num_data_blocks ||
    num_data_blocks % num_parity_blocks != 0
  ) {
        return XORResult::InvalidCounts;
  }

  if (
    reinterpret_cast<uintptr_t>((uintptr_t)data_buffer) % XOR_PTR_ALIGNMENT != 0 ||
    reinterpret_cast<uintptr_t>(parity_buffer) % XOR_PTR_ALIGNMENT != 0
  ) {
    return XORResult::InvalidAlignment;
  }

  std::memset(parity_buffer, 0, block_size * num_parity_blocks);

  for (unsigned i = 0; i < num_parity_blocks; ++i) {
    void * XOR_RESTRICT parity_block = reinterpret_cast<void*>(parity_buffer + i * block_size);
    for (unsigned j = i; j < num_data_blocks; j += num_parity_blocks) {
      const void * XOR_RESTRICT data_block = reinterpret_cast<const void*>(data_buffer + j * block_size);
      XOR_xor_blocks(parity_block, data_block, block_size);
    }
  }
  
  return XORResult::Success;  
}


XORResult xor_decode(
  uint8_t *XOR_RESTRICT data_buffer,
  const uint8_t *XOR_RESTRICT parity_buffer,
  uint32_t block_size,
  uint32_t num_data_blocks,
  uint32_t num_parity_blocks,
  std::bitset<256> block_bitmap
) {

  if ((block_bitmap & COMPLETE_DATA_BITMAP).count() == num_data_blocks) return XORResult::Success;

  if (block_size < XOR_MIN_BLOCK_SIZE || block_size % XOR_BLOCK_SIZE_MULTIPLE != 0) {
    return XORResult::InvalidSize;
  }

  if (
    num_data_blocks + num_parity_blocks > XOR_MAX_TOTAL_BLOCKS ||
    num_data_blocks < XOR_MIN_DATA_BLOCKS ||
    num_data_blocks > XOR_MAX_DATA_BLOCKS ||
    num_parity_blocks < XOR_MIN_PARITY_BLOCKS ||
    num_parity_blocks > XOR_MAX_PARITY_BLOCKS ||
    num_parity_blocks > num_data_blocks ||
    num_data_blocks % num_parity_blocks != 0
  ) {
        return XORResult::InvalidCounts;
  }

  if (
    reinterpret_cast<uintptr_t>((uintptr_t)data_buffer) % XOR_PTR_ALIGNMENT != 0 ||
    reinterpret_cast<uintptr_t>(parity_buffer) % XOR_PTR_ALIGNMENT != 0
  ) {
    return XORResult::InvalidAlignment;
  }



  std::bitset<128> lost_blocks;
  for (unsigned i = 0; i < num_parity_blocks; ++i) {
    if (!block_bitmap.test(128 + i)) lost_blocks.set(i);
  }
  
  
  for (unsigned j = 0; j < num_data_blocks; ++j) {
    uint32_t parity_idx = j % num_parity_blocks;
    if (!block_bitmap.test(j)) {
      if (lost_blocks.test(parity_idx)) return XORResult::DecodeFailure;
      lost_blocks.set(parity_idx);
    }
  }


  #if defined(TRY_XOR_AVX512)
    
  #elif defined(TRY_XOR_AVX2)
    // AVX2 implementation
  #elif defined(TRY_XOR_AVX)
    // AVX implementation
  #else
    for (unsigned i = 0; i < num_data_blocks; ++i) {
      if (block_bitmap.test(i)) continue;

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

  return XORResult::Success;
}