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

static uint64_t PREFETCH_SIZE = 4096; // 4096B


void xorec_init() {
  if (XOREC_INIT_CALLED) return;
  XOREC_INIT_CALLED = true;
  std::fill_n(COMPLETE_DATA_BITMAP.begin(), XOREC_MAX_DATA_BLOCKS, 1);
}

XorecResult xorec_encode(
  const uint8_t *XOREC_RESTRICT data_buffer,
  uint8_t *XOREC_RESTRICT parity_buffer,
  uint32_t block_size,
  uint32_t num_data_blocks,
  uint32_t num_parity_blocks,
  XorecVersion version
) {
  if (!XOREC_INIT_CALLED) throw_error("xorec_init() must be called before calling xorec_encode()");

  if (xorec_check_args(block_size, num_data_blocks, num_parity_blocks) != XorecResult::Success) return XorecResult::InvalidCounts;


  std::memset(parity_buffer, 0, block_size * num_parity_blocks);

  for (uint32_t i = 0; i < num_parity_blocks; ++i) {
    void * XOREC_RESTRICT parity_block = reinterpret_cast<void*>(parity_buffer + i * block_size);

    for (uint32_t j = i; j < num_data_blocks; j += num_parity_blocks) {
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
  if (!XOREC_INIT_CALLED) throw_error("xorec_init() must be called before calling xorec_decode()");
  if (!recovery_needed(block_bitmap)) return XorecResult::Success;

  if (xorec_check_args(block_size, num_data_blocks, num_parity_blocks) != XorecResult::Success) return XorecResult::InvalidCounts;

  if (!is_recoverable(block_bitmap, num_data_blocks, num_parity_blocks)) return XorecResult::DecodeFailure;

  for (uint32_t i = 0; i < num_data_blocks; ++i) {
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
    for (uint32_t j = i % num_parity_blocks; j < num_data_blocks; j += num_parity_blocks) {
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





XorecResult xorec_pipelined_encode(
  const uint8_t *XOREC_RESTRICT data_buffer,  // unified memory
  uint8_t *XOREC_RESTRICT parity_buffer,      // host memory
  uint32_t block_size,
  uint32_t num_data_blocks,
  uint32_t num_parity_blocks,
  XorecVersion version
) {
  if (!XOREC_INIT_CALLED) throw_error("xorec_init() must be called before calling xorec_pipelined_encode()");
  if (xorec_check_args(block_size, num_data_blocks, num_parity_blocks) != XorecResult::Success) return XorecResult::InvalidCounts;
  
  uint64_t prefetched_bytes = 0;
  cudaStream_t prefetch_stream;
  bool do_synchronize = true;

  cudaError_t err = cudaStreamCreate(&prefetch_stream);
  if (err != cudaSuccess) throw_error("cudaStreamCreate() failed: " + std::string(cudaGetErrorString(err)));

  err = cudaMemPrefetchAsync(data_buffer, PREFETCH_SIZE, cudaCpuDeviceId, prefetch_stream);
  if (err != cudaSuccess) throw_error("cudaMemPrefetchAsync() failed: " + std::string(cudaGetErrorString(err)));
  

  std::memset(parity_buffer, 0, block_size * num_parity_blocks);

  for (uint32_t i = 0; i < num_data_blocks; ++i) {

    if (do_synchronize) { // synchronize previous prefetch
      err = cudaStreamSynchronize(prefetch_stream);
      if (err != cudaSuccess) throw_error("cudaStreamSynchronize() failed: " + std::string(cudaGetErrorString(err)));
      prefetched_bytes += PREFETCH_SIZE;
      err = cudaStreamDestroy(prefetch_stream);
      if (err != cudaSuccess) throw_error("cudaStreamDestroy() failed: " + std::string(cudaGetErrorString(err)));
      do_synchronize = false;
    }

    uint32_t curr_byte = i * block_size;
    if (curr_byte >= prefetched_bytes - PREFETCH_SIZE) { // prefetch if needed
      uint32_t prefetch_amt =  (prefetched_bytes + PREFETCH_SIZE < num_data_blocks * block_size) ? PREFETCH_SIZE : num_data_blocks * block_size - prefetched_bytes;
      
      err = cudaStreamCreate(&prefetch_stream);
      if (err != cudaSuccess) throw_error("cudaStreamCreate() failed: " + std::string(cudaGetErrorString(err)));

      err = cudaMemPrefetchAsync(data_buffer + prefetched_bytes, prefetch_amt, cudaCpuDeviceId, prefetch_stream);
      if (err != cudaSuccess) throw_error("cudaMemPrefetchAsync() failed: " + std::string(cudaGetErrorString(err)));
      do_synchronize = true;
    }

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



XorecResult xorec_pipeline_decode(
  uint8_t *XOREC_RESTRICT data_buffer,
  const uint8_t *XOREC_RESTRICT parity_buffer,
  uint32_t block_size,
  uint32_t num_data_blocks,
  uint32_t num_parity_blocks,
  const uint8_t * XOREC_RESTRICT block_bitmap,
  XorecVersion version
);