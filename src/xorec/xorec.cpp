/**
 * @file xorec.cpp
 * @brief Implements XOR-based erasure encoding and decoding functions.
 */


#include "xorec.h"
#include <cstring>
#include <iostream>


XorecResult xorec_encode(
  const uint8_t *XOREC_RESTRICT data_buffer,
  uint8_t *XOREC_RESTRICT parity_buffer,
  uint32_t block_size,
  uint32_t num_data_blocks,
  uint32_t num_parity_blocks,
  XorecVersion version
) {
  if (xorec_check_args(block_size, num_data_blocks, num_parity_blocks) != XorecResult::Success) return XorecResult::InvalidCounts;


  std::memset(parity_buffer, 0, block_size * num_parity_blocks);

  for (unsigned i = 0; i < num_parity_blocks; ++i) {
    void * XOREC_RESTRICT parity_block = reinterpret_cast<void*>(parity_buffer + i * block_size);

    for (unsigned j = i; j < num_data_blocks; j += num_parity_blocks) {
      const void * XOREC_RESTRICT data_block = reinterpret_cast<const void*>(data_buffer + j * block_size);
      switch (version) {
        case XorecVersion::Scalar:
          xorec_xor_blocks_scalar(parity_block, data_block, block_size);
          break;
        case XorecVersion::AVX:
          xorec_xor_blocks_avx(parity_block, data_block, block_size);
          break;
        case XorecVersion::AVX2:
          xorec_xor_blocks_avx2(parity_block, data_block, block_size);
          break;
      }
    }
  }
  
  return XorecResult::Success;  
}


XorecResult xorec_decode(
  uint8_t *XOREC_RESTRICT data_buffer,
  const uint8_t *XOREC_RESTRICT parity_buffer,
  uint32_t block_size,
  uint32_t num_data_blocks,
  uint32_t num_parity_blocks,
  const uint8_t * XOREC_RESTRICT block_bitmap,
  XorecVersion version
) {
  std::array<uint8_t, XOREC_MAX_DATA_BLOCKS> temp = {0};
  and_bitmap(temp.data(), block_bitmap, COMPLETE_DATA_BITMAP.data(), XOREC_MAX_DATA_BLOCKS);

  if (bit_count(temp.data(), XOREC_MAX_DATA_BLOCKS) == num_data_blocks) return XorecResult::Success;

  if (xorec_check_args(block_size, num_data_blocks, num_parity_blocks) != XorecResult::Success) return XorecResult::InvalidCounts;

  std::array<uint8_t, XOREC_MAX_PARITY_BLOCKS> lost_blocks = {0}; // indicate for each parity block if recovery has to happen

  for (unsigned i = 0; i < num_parity_blocks; ++i) {
    if (block_bitmap[XOREC_MAX_PARITY_BLOCKS + i]) lost_blocks[i] = 1;
  }
  
  
  for (unsigned j = 0; j < num_data_blocks; ++j) {
    uint32_t parity_idx = j % num_parity_blocks;
    if (!block_bitmap[j]) {
      if (lost_blocks[parity_idx]) return XorecResult::DecodeFailure;
      lost_blocks[parity_idx] = 1;
    }
  }

  for (unsigned i = 0; i < num_data_blocks; ++i) {
    if (block_bitmap[i]) continue;

    void * XOREC_RESTRICT recover_block = reinterpret_cast<void*>(data_buffer + i * block_size);
    const void * XOREC_RESTRICT parity_block = reinterpret_cast<const void*>(parity_buffer + (i % num_parity_blocks) * block_size);

    // Copy the parity block to the recover block
    switch (version) {
      case XorecVersion::Scalar:
        xorec_copy_blocks_scalar(recover_block, parity_block, block_size);
        break;
      case XorecVersion::AVX:
        xorec_copy_blocks_avx(recover_block, parity_block, block_size);
        break;
      case XorecVersion::AVX2:
        xorec_copy_blocks_avx2(recover_block, parity_block, block_size);
        break;
    }

    // XOR the recover block with the other data blocks
    for (unsigned j = i % num_parity_blocks; j < num_data_blocks; j += num_parity_blocks) {
      if (i == j) continue;

      const void * XOREC_RESTRICT data_block = reinterpret_cast<const void*>(data_buffer + j * block_size);
      switch (version) {
        case XorecVersion::Scalar:
          xorec_xor_blocks_scalar(recover_block, data_block, block_size);
          break;
        case XorecVersion::AVX:
          xorec_xor_blocks_avx(recover_block, data_block, block_size);
          break;
        case XorecVersion::AVX2:
          xorec_xor_blocks_avx2(recover_block, data_block, block_size);
          break;
      }
    }
  }

  return XorecResult::Success;
}