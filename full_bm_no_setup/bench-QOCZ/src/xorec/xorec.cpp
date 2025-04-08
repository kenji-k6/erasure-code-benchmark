/**
 * @file xorec.cpp
 * @brief Implements XOR-based erasure encoding and decoding functions.
 */

 #include "xorec.hpp"
 #include <cstring>
 #include "utils.hpp"
 
static bool XOREC_INIT_CALLED = false;



void xorec_init(size_t num_data_blocks, [[maybe_unused]] size_t num_parity_blocks) {
  if (XOREC_INIT_CALLED) return;
  XOREC_INIT_CALLED = true;
  std::fill_n(COMPLETE_DATA_BITMAP.begin(), num_data_blocks, 1);
}

XorecResult xorec_encode_avx(
  const uint8_t *XOREC_RESTRICT data_buffer,
  uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks
) {
  if (!XOREC_INIT_CALLED) throw_error("xorec_init() must be called before calling xorec_encode()");

  if (xorec_check_args(block_size, num_data_blocks, num_parity_blocks) != XorecResult::Success) return XorecResult::InvalidCounts;

  unsigned i;

  for (i = 0; i < num_parity_blocks; ++i) {
    void * XOREC_RESTRICT parity_block = reinterpret_cast<void*>(parity_buffer + i * block_size);
    const void * XOREC_RESTRICT data_block = reinterpret_cast<const void*>(data_buffer + i * block_size);
    memcpy(parity_block, data_block, block_size);
  }

  for (; i < num_data_blocks; ++i) {
    void * XOREC_RESTRICT parity_block = reinterpret_cast<void*>(parity_buffer + (i % num_parity_blocks) * block_size);
    const void * XOREC_RESTRICT data_block = reinterpret_cast<const void*>(data_buffer + i * block_size);
    
    xorec_xor_blocks_avx(parity_block, data_block, block_size);
  }  
  return XorecResult::Success;  
}


XorecResult xorec_decode_avx(
  uint8_t *XOREC_RESTRICT data_buffer,
  const uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  const uint8_t * XOREC_RESTRICT block_bitmap
) {
  if (!XOREC_INIT_CALLED) throw_error("xorec_init() must be called before calling xorec_decode()");
  if (!recovery_needed(block_bitmap)) return XorecResult::Success;

  if (xorec_check_args(block_size, num_data_blocks, num_parity_blocks) != XorecResult::Success) return XorecResult::InvalidCounts;

  if (!is_recoverable(block_bitmap, num_data_blocks, num_parity_blocks)) return XorecResult::DecodeFailure;

  for (unsigned i = 0; i < num_data_blocks; ++i) {
    if (block_bitmap[i]) continue;

    void * XOREC_RESTRICT recover_block = reinterpret_cast<void*>(data_buffer + i * block_size);
    const void * XOREC_RESTRICT parity_block = reinterpret_cast<const void*>(parity_buffer + (i % num_parity_blocks) * block_size);

    // Copy the parity block to the recover block
    memcpy(recover_block, parity_block, block_size);

    // XOR the recover block with the other data blocks
    for (unsigned j = i % num_parity_blocks; j < num_data_blocks; j += num_parity_blocks) {
      if (i == j) continue;

      const void * XOREC_RESTRICT data_block = reinterpret_cast<const void*>(data_buffer + j * block_size);
      xorec_xor_blocks_avx(recover_block, data_block, block_size);
    }
  }
  return XorecResult::Success;
}





XorecResult xorec_encode_avx2(
  const uint8_t *XOREC_RESTRICT data_buffer,
  uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks
) {
  if (!XOREC_INIT_CALLED) throw_error("xorec_init() must be called before calling xorec_encode()");

  if (xorec_check_args(block_size, num_data_blocks, num_parity_blocks) != XorecResult::Success) return XorecResult::InvalidCounts;

  unsigned i;

  for (i = 0; i < num_parity_blocks; ++i) {
    void * XOREC_RESTRICT parity_block = reinterpret_cast<void*>(parity_buffer + i * block_size);
    const void * XOREC_RESTRICT data_block = reinterpret_cast<const void*>(data_buffer + i * block_size);
    memcpy(parity_block, data_block, block_size);
  }

  for (; i < num_data_blocks; ++i) {
    void * XOREC_RESTRICT parity_block = reinterpret_cast<void*>(parity_buffer + (i % num_parity_blocks) * block_size);
    const void * XOREC_RESTRICT data_block = reinterpret_cast<const void*>(data_buffer + i * block_size);
    
    xorec_xor_blocks_avx2(parity_block, data_block, block_size);
  }  
  return XorecResult::Success;  
}


XorecResult xorec_decode_avx2(
  uint8_t *XOREC_RESTRICT data_buffer,
  const uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  const uint8_t * XOREC_RESTRICT block_bitmap
) {
  if (!XOREC_INIT_CALLED) throw_error("xorec_init() must be called before calling xorec_decode()");
  if (!recovery_needed(block_bitmap)) return XorecResult::Success;

  if (xorec_check_args(block_size, num_data_blocks, num_parity_blocks) != XorecResult::Success) return XorecResult::InvalidCounts;

  if (!is_recoverable(block_bitmap, num_data_blocks, num_parity_blocks)) return XorecResult::DecodeFailure;

  for (unsigned i = 0; i < num_data_blocks; ++i) {
    if (block_bitmap[i]) continue;

    void * XOREC_RESTRICT recover_block = reinterpret_cast<void*>(data_buffer + i * block_size);
    const void * XOREC_RESTRICT parity_block = reinterpret_cast<const void*>(parity_buffer + (i % num_parity_blocks) * block_size);

    // Copy the parity block to the recover block
    memcpy(recover_block, parity_block, block_size);

    // XOR the recover block with the other data blocks
    for (unsigned j = i % num_parity_blocks; j < num_data_blocks; j += num_parity_blocks) {
      if (i == j) continue;

      const void * XOREC_RESTRICT data_block = reinterpret_cast<const void*>(data_buffer + j * block_size);
      xorec_xor_blocks_avx2(recover_block, data_block, block_size);
    }
  }
  return XorecResult::Success;
}





XorecResult xorec_encode_avx512(
  const uint8_t *XOREC_RESTRICT data_buffer,
  uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks
) {
  #ifdef __AVX512F__
  if (!XOREC_INIT_CALLED) throw_error("xorec_init() must be called before calling xorec_encode()");

  if (xorec_check_args(block_size, num_data_blocks, num_parity_blocks) != XorecResult::Success) return XorecResult::InvalidCounts;

  unsigned i;

  for (i = 0; i < num_parity_blocks; ++i) {
    void * XOREC_RESTRICT parity_block = reinterpret_cast<void*>(parity_buffer + i * block_size);
    const void * XOREC_RESTRICT data_block = reinterpret_cast<const void*>(data_buffer + i * block_size);
    memcpy(parity_block, data_block, block_size);
  }

  for (; i < num_data_blocks; ++i) {
    void * XOREC_RESTRICT parity_block = reinterpret_cast<void*>(parity_buffer + (i % num_parity_blocks) * block_size);
    const void * XOREC_RESTRICT data_block = reinterpret_cast<const void*>(data_buffer + i * block_size);
    
    xorec_xor_blocks_avx512(parity_block, data_block, block_size);
  }
  return XorecResult::Success; 
  #else
  throw_error("AVX512 is not supported on this platform");
  #endif
}


XorecResult xorec_decode_avx512(
  uint8_t *XOREC_RESTRICT data_buffer,
  const uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  const uint8_t * XOREC_RESTRICT block_bitmap
) {
  #ifdef __AVX512F__
  if (!XOREC_INIT_CALLED) throw_error("xorec_init() must be called before calling xorec_decode()");
  if (!recovery_needed(block_bitmap)) return XorecResult::Success;

  if (xorec_check_args(block_size, num_data_blocks, num_parity_blocks) != XorecResult::Success) return XorecResult::InvalidCounts;

  if (!is_recoverable(block_bitmap, num_data_blocks, num_parity_blocks)) return XorecResult::DecodeFailure;

  for (unsigned i = 0; i < num_data_blocks; ++i) {
    if (block_bitmap[i]) continue;

    void * XOREC_RESTRICT recover_block = reinterpret_cast<void*>(data_buffer + i * block_size);
    const void * XOREC_RESTRICT parity_block = reinterpret_cast<const void*>(parity_buffer + (i % num_parity_blocks) * block_size);

    // Copy the parity block to the recover block
    memcpy(recover_block, parity_block, block_size);

    // XOR the recover block with the other data blocks
    for (unsigned j = i % num_parity_blocks; j < num_data_blocks; j += num_parity_blocks) {
      if (i == j) continue;

      const void * XOREC_RESTRICT data_block = reinterpret_cast<const void*>(data_buffer + j * block_size);
      xorec_xor_blocks_avx512(recover_block, data_block, block_size);
    }
  }
  return XorecResult::Success;
  #else
  throw_error("AVX512 is not supported on this platform");
  #endif
}