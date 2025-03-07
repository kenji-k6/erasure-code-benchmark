/**
 * @file xorec.cpp
 * @brief Implements XOR-based erasure encoding and decoding functions.
 */


#include "xorec.h"
#include <cstring>
#include <iostream>


XORResult xor_encode(
  const uint8_t *XOR_RESTRICT data_buffer,
  uint8_t *XOR_RESTRICT parity_buffer,
  uint32_t block_size,
  uint32_t num_data_blocks,
  uint32_t num_parity_blocks,
  XORVersion version
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
      switch (version) {
        case XORVersion::Auto:
          XOR_xor_blocks(parity_block, data_block, block_size);
          break;
        case XORVersion::Scalar:
          XOR_xor_blocks_scalar(parity_block, data_block, block_size);
          break;
        case XORVersion::ScalarNoOpt:
          XOR_xor_blocks_scalar_no_opt(parity_block, data_block, block_size);
          break;
        case XORVersion::AVX:
          XOR_xor_blocks_avx(parity_block, data_block, block_size);
          break;
        case XORVersion::AVX2:
          XOR_xor_blocks_avx2(parity_block, data_block, block_size);
          break;
      }
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
  std::bitset<256> block_bitmap,
  XORVersion version
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

  for (unsigned i = 0; i < num_data_blocks; ++i) {
    if (block_bitmap.test(i)) continue;

    void * XOR_RESTRICT recover_block = reinterpret_cast<void*>(data_buffer + i * block_size);
    const void * XOR_RESTRICT parity_block = reinterpret_cast<const void*>(parity_buffer + (i % num_parity_blocks) * block_size);

    // Copy the parity block to the recover block
    switch (version) {
      case XORVersion::Auto:
        XOR_copy_blocks(recover_block, parity_block, block_size);
        break;
      case XORVersion::Scalar:
        XOR_copy_blocks_scalar(recover_block, parity_block, block_size);
        break;
      case XORVersion::ScalarNoOpt:
        XOR_copy_blocks_scalar(recover_block, parity_block, block_size);
        break;
      case XORVersion::AVX:
        XOR_copy_blocks_avx(recover_block, parity_block, block_size);
        break;
      case XORVersion::AVX2:
        XOR_copy_blocks_avx2(recover_block, parity_block, block_size);
        break;
    }

    // XOR the recover block with the other data blocks
    for (unsigned j = i % num_parity_blocks; j < num_data_blocks; j += num_parity_blocks) {
      if (i == j) continue;

      const void * XOR_RESTRICT data_block = reinterpret_cast<const void*>(data_buffer + j * block_size);
      switch (version) {
        case XORVersion::Auto:
          XOR_xor_blocks(recover_block, data_block, block_size);
          break;
        case XORVersion::Scalar:
          XOR_xor_blocks_scalar(recover_block, data_block, block_size);
          break;
        case XORVersion::ScalarNoOpt:
          XOR_xor_blocks_scalar_no_opt(recover_block, data_block, block_size);
          break;
        case XORVersion::AVX:
          XOR_xor_blocks_avx(recover_block, data_block, block_size);
          break;
        case XORVersion::AVX2:
          XOR_xor_blocks_avx2(recover_block, data_block, block_size);
          break;
      }
    }
  }

  return XORResult::Success;
}