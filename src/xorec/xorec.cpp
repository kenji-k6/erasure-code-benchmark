/**
 * @file xorec.cpp
 * @brief Implements XOR-based erasure encoding and decoding functions.
 */


#include "xorec.hpp"
#include <cstring>
#include <iostream>
#include "utils.hpp"
#include <cuda_runtime.h>

static bool XOREC_INIT_CALLED = false;



void xorec_init() {
  if (XOREC_INIT_CALLED) return;
  XOREC_INIT_CALLED = true;
  std::fill_n(COMPLETE_DATA_BITMAP.begin(), XOREC_MAX_DATA_BLOCKS, 1);
}

XorecResult xorec_encode(
  const uint8_t *XOREC_RESTRICT data_buffer,
  uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  XorecVersion version
) {
  if (!XOREC_INIT_CALLED) throw_error("xorec_init() must be called before calling xorec_encode()");

  if (xorec_check_args(block_size, num_data_blocks, num_parity_blocks) != XorecResult::Success) return XorecResult::InvalidCounts;


  std::memset(parity_buffer, 0, block_size * num_parity_blocks);

  for (unsigned i = 0; i < num_data_blocks; ++i) {
    void * XOREC_RESTRICT parity_block = reinterpret_cast<void*>(parity_buffer + (i % num_parity_blocks) * block_size);
    const void * XOREC_RESTRICT data_block = reinterpret_cast<const void*>(data_buffer + i * block_size);

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
  return XorecResult::Success;  
}


XorecResult xorec_decode(
  uint8_t *XOREC_RESTRICT data_buffer,
  const uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  const uint8_t * XOREC_RESTRICT block_bitmap,
  XorecVersion version
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


XorecResult xorec_prefetch_encode(
  const uint8_t *XOREC_RESTRICT data_buffer,  // unified memory
  uint8_t *XOREC_RESTRICT parity_buffer,      // host memory
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  size_t prefetch_bytes,
  XorecVersion version
) {
  
  if (!XOREC_INIT_CALLED) throw_error("xorec_init() must be called before calling xorec_pipelined_encode()");
  if (xorec_check_args(block_size, num_data_blocks, num_parity_blocks) != XorecResult::Success) return XorecResult::InvalidCounts;

  size_t prefetch_blocks = prefetch_bytes / block_size;
  if (prefetch_blocks == 0) prefetch_blocks = 1;

  cudaStream_t prefetch_stream;
  cudaStreamCreate(&prefetch_stream);


  size_t initial_prefetch_bytes = std::min(prefetch_blocks * block_size, num_data_blocks * block_size);
  cudaMemPrefetchAsync(data_buffer, initial_prefetch_bytes, cudaCpuDeviceId, prefetch_stream);
  
  std::memset(parity_buffer, 0, block_size * num_parity_blocks);



  for (unsigned i = 0; i < num_data_blocks; ++i) {
    if (i % prefetch_blocks == 0) { // We are at a prefetch interval
      cudaStreamSynchronize(prefetch_stream);

      if (i + prefetch_blocks < num_data_blocks) {
        size_t remaining_blocks = num_data_blocks - (i + prefetch_blocks);
        size_t prefetch_blks = std::min(prefetch_blocks, remaining_blocks);
        cudaMemPrefetchAsync(data_buffer + (i+prefetch_blocks) * block_size, prefetch_blks * block_size, cudaCpuDeviceId, prefetch_stream);
      }
    }

    void * XOREC_RESTRICT parity_block = reinterpret_cast<void*>(parity_buffer + (i % num_parity_blocks) * block_size);
    const void * XOREC_RESTRICT data_block = reinterpret_cast<const void*>(data_buffer + i * block_size);

    switch(version) {
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

  cudaStreamDestroy(prefetch_stream);


  return XorecResult::Success;
}



XorecResult xorec_prefetch_decode(
  uint8_t *XOREC_RESTRICT data_buffer,
  uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  const uint8_t * XOREC_RESTRICT block_bitmap,
  size_t prefetch_bytes,
  XorecVersion version
) {

  std::array<uint8_t, XOREC_MAX_PARITY_BLOCKS> restore_parity_bitmap = {0}; // Bitmap to track which parity blocks need to be restored
  if (!XOREC_INIT_CALLED) throw_error("xorec_init() must be called before calling xorec_decode()");
  if (!recovery_needed(block_bitmap)) return XorecResult::Success;
  if (xorec_check_args(block_size, num_data_blocks, num_parity_blocks) != XorecResult::Success) return XorecResult::InvalidCounts;
  if (!is_recoverable(block_bitmap, num_data_blocks, num_parity_blocks)) return XorecResult::DecodeFailure;

  for (unsigned i = 0; i < num_data_blocks; ++i) {
    if (!block_bitmap[i]) restore_parity_bitmap[i % num_parity_blocks] = 1;
  }


  size_t prefetch_blocks = prefetch_bytes / block_size;

  if (prefetch_blocks == 0) prefetch_blocks = 1;

  cudaStream_t prefetch_stream;
  cudaStreamCreate(&prefetch_stream);

  size_t initial_prefetch_bytes = std::min(prefetch_blocks * block_size, num_data_blocks * block_size);
  cudaMemPrefetchAsync(data_buffer, initial_prefetch_bytes, cudaCpuDeviceId, prefetch_stream);

  for (unsigned i = 0; i < num_data_blocks; ++i) {
    if (i % prefetch_blocks == 0) { // We are at a prefetch interval
      cudaStreamSynchronize(prefetch_stream);

      if (i + prefetch_blocks < num_data_blocks) {
        size_t remaining_blocks = num_data_blocks - (i + prefetch_blocks);
        size_t prefetch_blks = std::min(prefetch_blocks, remaining_blocks);
        cudaMemPrefetchAsync(data_buffer + (i+prefetch_blocks) * block_size, prefetch_blks * block_size, cudaCpuDeviceId, prefetch_stream);
      }
    }

    if (restore_parity_bitmap[i % num_parity_blocks] && block_bitmap[i]) {
      void * XOREC_RESTRICT curr_block = reinterpret_cast<void*>(data_buffer + i * block_size);
      void * XOREC_RESTRICT parity_block = reinterpret_cast<void*>(parity_buffer + (i % num_parity_blocks) * block_size);

      switch(version) {
        case XorecVersion::Scalar:
          xorec_xor_blocks_scalar(parity_block, curr_block, block_size);
          break;
        case XorecVersion::AVX:
          xorec_xor_blocks_avx(parity_block, curr_block, block_size);
          break;
        case XorecVersion::AVX2:
          xorec_xor_blocks_avx2(parity_block, curr_block, block_size);
          break;
      }
    }
  }

  for (unsigned i = 0; i < num_data_blocks; ++i) {
    if (!block_bitmap[i]) {
      memcpy(data_buffer + i * block_size, parity_buffer + (i % num_parity_blocks) * block_size, block_size);
    }
  }

  return XorecResult::Success;
}