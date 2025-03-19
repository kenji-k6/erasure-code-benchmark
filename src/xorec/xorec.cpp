/**
 * @file xorec.cpp
 * @brief Implements XOR-based erasure encoding and decoding functions.
 */


 #include "xorec.hpp"
 #include <cstring>
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

  unsigned i;

  for (i = 0; i < num_parity_blocks; ++i) {
    void * XOREC_RESTRICT parity_block = reinterpret_cast<void*>(parity_buffer + i * block_size);
    const void * XOREC_RESTRICT data_block = reinterpret_cast<const void*>(data_buffer + i * block_size);
    memcpy(parity_block, data_block, block_size);
  }

  for (; i < num_data_blocks; ++i) {
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


XorecResult xorec_unified_prefetch_encode(
  const uint8_t *XOREC_RESTRICT data_buffer,  // unified memory
  uint8_t *XOREC_RESTRICT parity_buffer,      // host memory
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  XorecVersion version
) {
  cudaMemPrefetchAsync(data_buffer, num_data_blocks * block_size, cudaCpuDeviceId);
  if (!XOREC_INIT_CALLED) throw_error("xorec_init() must be called before calling xorec_unified_prefetch_encode()");
  if (xorec_check_args(block_size, num_data_blocks, num_parity_blocks) != XorecResult::Success) return XorecResult::InvalidCounts;

  std::memset(parity_buffer, 0, block_size * num_parity_blocks);
  cudaDeviceSynchronize();

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



XorecResult xorec_unified_prefetch_decode(
  uint8_t *XOREC_RESTRICT data_buffer,
  const uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  const uint8_t * XOREC_RESTRICT block_bitmap,
  XorecVersion version
) {

  cudaMemPrefetchAsync(data_buffer, num_data_blocks * block_size, cudaCpuDeviceId);

  if (!XOREC_INIT_CALLED) throw_error("xorec_init() must be called before calling xorec_unified_prefetch_decode()");
  if (!recovery_needed(block_bitmap)) return XorecResult::Success;
  if (xorec_check_args(block_size, num_data_blocks, num_parity_blocks) != XorecResult::Success) return XorecResult::InvalidCounts;
  if (!is_recoverable(block_bitmap, num_data_blocks, num_parity_blocks)) return XorecResult::DecodeFailure;

  cudaDeviceSynchronize();
  for (unsigned i = 0; i < num_data_blocks; ++i) {
    if (block_bitmap[i]) continue;

    void * XOREC_RESTRICT recover_block = reinterpret_cast<void*>(data_buffer + i * block_size);
    const void * XOREC_RESTRICT parity_block = reinterpret_cast<const void*>(parity_buffer + (i % num_parity_blocks) * block_size);

    // Copy the parity block to the recover block
    cudaMemcpy(recover_block, parity_block, block_size, cudaMemcpyHostToHost);

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






XorecResult xorec_gpu_prefetch_encode(
  const uint8_t *XOREC_RESTRICT gpu_data_buffer,
  uint8_t *XOREC_RESTRICT cpu_data_buffer,
  uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  size_t prefetch_bytes,
  XorecVersion version
) {
  cudaMemcpyAsync(cpu_data_buffer, gpu_data_buffer, num_data_blocks * block_size, cudaMemcpyDeviceToHost);
  if (!XOREC_INIT_CALLED) throw_error("xorec_init() must be called before calling xorec_gpu_prefetch_encode()");
  if (xorec_check_args(block_size, num_data_blocks, num_parity_blocks) != XorecResult::Success) return XorecResult::InvalidCounts;

  std::memset(parity_buffer, 0, block_size * num_parity_blocks);

  cudaDeviceSynchronize();

  for (unsigned i = 0; i < num_data_blocks; ++i) {
    void * XOREC_RESTRICT parity_block = reinterpret_cast<void*>(parity_buffer + (i % num_parity_blocks) * block_size);
    const void * XOREC_RESTRICT data_block = reinterpret_cast<const void*>(cpu_data_buffer + i * block_size);

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


/**
 * @brief Decodes data using XOR-based erasure coding.
 * 
 * @param data_buffer Pointer to the data buffer.
 * @param parity_buffer Pointer to the parity buffer.
 * @param block_size Size of each block in bytes.
 * @param num_data_blocks Number of data blocks.
 * @param num_parity_blocks Number of parity blocks.
 * @param block_bitmap A bitmap indicating which blocks are present.
 * @return XorecResult XorecResult indicating success or failure.
 */
XorecResult xorec_gpu_prefetch_decode(
  uint8_t *XOREC_RESTRICT gpu_data_buffer,
  uint8_t *XOREC_RESTRICT cpu_data_buffer,
  const uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  const uint8_t * XOREC_RESTRICT block_bitmap,
  size_t prefetch_bytes,
  XorecVersion version
) {
  cudaMemcpyAsync(cpu_data_buffer, gpu_data_buffer, num_data_blocks * block_size, cudaMemcpyDeviceToHost);


  if (!XOREC_INIT_CALLED) throw_error("xorec_init() must be called before calling xorec_decode()");
  if (!recovery_needed(block_bitmap)) return XorecResult::Success;
  if (xorec_check_args(block_size, num_data_blocks, num_parity_blocks) != XorecResult::Success) return XorecResult::InvalidCounts;
  if (!is_recoverable(block_bitmap, num_data_blocks, num_parity_blocks)) return XorecResult::DecodeFailure;
  int num_lost_blocks = 0;

  cudaDeviceSynchronize();

  for (unsigned i = 0; i < num_data_blocks; ++i) {
    if (block_bitmap[i]) continue;
    void * XOREC_RESTRICT recover_block = reinterpret_cast<void*>(cpu_data_buffer + i * block_size);
    const void * XOREC_RESTRICT parity_block = reinterpret_cast<const void*>(parity_buffer + (i % num_parity_blocks) * block_size);

    cudaMemcpy(recover_block, parity_block, block_size, cudaMemcpyHostToHost);

    for (unsigned j = i % num_parity_blocks; j < num_data_blocks; j += num_parity_blocks) {
      if (i == j) continue;

      const void * XOREC_RESTRICT data_block = reinterpret_cast<const void*>(cpu_data_buffer + j * block_size);
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
    
    cudaMemcpyAsync(gpu_data_buffer + i * block_size, recover_block, block_size, cudaMemcpyHostToDevice);
  }
  return XorecResult::Success;
}