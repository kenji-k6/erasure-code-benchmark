/**
 * @file xorec.cpp
 * @brief Implements XOR-based erasure encoding and decoding functions.
 */


 #include "xorec.hpp"
 #include <cstring>
 #include "utils.hpp"
 #include <cuda_runtime.h>
 
static bool XOREC_INIT_CALLED = false;
std::vector<uint8_t> COMPLETE_DATA_BITMAP = {};


void xorec_init(size_t num_data_blocks) {
  if (XOREC_INIT_CALLED) return;

  COMPLETE_DATA_BITMAP.resize(num_data_blocks);
  std::fill_n(COMPLETE_DATA_BITMAP.begin(), num_data_blocks, 1);
  XOREC_INIT_CALLED = true;
}

XorecResult xorec_encode(
  const uint8_t *XOREC_RESTRICT data_buf,
  uint8_t *XOREC_RESTRICT parity_buf,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  XorecVersion version
) {
  if (!XOREC_INIT_CALLED) throw_error("xorec_init() must be called before calling xorec_encode()");

  XorecResult err = xorec_check_args(data_buf, parity_buf, block_size, num_data_blocks, num_parity_blocks);
  if (err != XorecResult::Success) return err;

  memcpy(parity_buf, data_buf, num_parity_blocks*block_size);

  for (unsigned i = num_parity_blocks; i < num_data_blocks; ++i) {
    void * XOREC_RESTRICT parity_block = reinterpret_cast<void*>(parity_buf + (i % num_parity_blocks) * block_size);
    const void * XOREC_RESTRICT data_block = reinterpret_cast<const void*>(data_buf + i * block_size);

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
      case XorecVersion::AVX512:
        xorec_xor_blocks_avx512(parity_block, data_block, block_size);
        break;
    }
  }  
  return XorecResult::Success;  
}


XorecResult xorec_decode(
  uint8_t *XOREC_RESTRICT data_buf,
  const uint8_t *XOREC_RESTRICT parity_buf,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  const uint8_t * XOREC_RESTRICT block_bitmap,
  XorecVersion version
) {
  if (!XOREC_INIT_CALLED) throw_error("xorec_init() must be called before calling xorec_decode()");

  XorecResult err = xorec_check_args(data_buf, parity_buf, block_size, num_data_blocks, num_parity_blocks);
  if (err != XorecResult::Success) return err;

  if (!require_recovery(num_data_blocks, block_bitmap)) return XorecResult::Success;
  if (!is_recoverable(num_data_blocks, num_parity_blocks, block_bitmap)) return XorecResult::DecodeFailure;

  for (unsigned i = 0; i < num_data_blocks; ++i) {
    if (block_bitmap[i]) continue;

    void * XOREC_RESTRICT recover_block = reinterpret_cast<void*>(data_buf + i * block_size);
    const void * XOREC_RESTRICT parity_block = reinterpret_cast<const void*>(parity_buf + (i % num_parity_blocks) * block_size);

    // Copy the parity block to the recover block
    memcpy(recover_block, parity_block, block_size);

    // XOR the recover block with the other data blocks
    for (unsigned j = i % num_parity_blocks; j < num_data_blocks; j += num_parity_blocks) {
      if (i == j) continue;

      const void * XOREC_RESTRICT data_block = reinterpret_cast<const void*>(data_buf + j * block_size);
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
        case XorecVersion::AVX512:
          xorec_xor_blocks_avx512(recover_block, data_block, block_size);
          break;
      }
    }
  }

  return XorecResult::Success;
}


XorecResult xorec_unified_prefetch_encode(
  const uint8_t *XOREC_RESTRICT data_buf,  // unified memory
  uint8_t *XOREC_RESTRICT parity_buf,      // host memory
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  XorecVersion version
) {
  cudaMemPrefetchAsync(data_buf, num_data_blocks * block_size, cudaCpuDeviceId);
  if (!XOREC_INIT_CALLED) throw_error("xorec_init() must be called before calling xorec_unified_prefetch_encode()");

  XorecResult err = xorec_check_args(data_buf, parity_buf, block_size, num_data_blocks, num_parity_blocks);
  if (err != XorecResult::Success) return err;

  cudaMemcpyAsync(parity_buf, data_buf, num_parity_blocks * block_size, cudaMemcpyHostToHost);
  cudaDeviceSynchronize();

  for (unsigned i = num_parity_blocks; i < num_data_blocks; ++i) {
    void * XOREC_RESTRICT parity_block = reinterpret_cast<void*>(parity_buf + (i % num_parity_blocks) * block_size);
    const void * XOREC_RESTRICT data_block = reinterpret_cast<const void*>(data_buf + i * block_size);

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
      case XorecVersion::AVX512:
        xorec_xor_blocks_avx512(parity_block, data_block, block_size);
        break;
    }
  }

  return XorecResult::Success;
}



XorecResult xorec_unified_prefetch_decode(
  uint8_t *XOREC_RESTRICT data_buf,
  const uint8_t *XOREC_RESTRICT parity_buf,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  const uint8_t * XOREC_RESTRICT block_bitmap,
  XorecVersion version
) {

  cudaMemPrefetchAsync(data_buf, num_data_blocks * block_size, cudaCpuDeviceId);

  if (!XOREC_INIT_CALLED) throw_error("xorec_init() must be called before calling xorec_unified_prefetch_decode()");

  XorecResult err = xorec_check_args(data_buf, parity_buf, block_size, num_data_blocks, num_parity_blocks);
  if (err != XorecResult::Success) return err;

  if (!require_recovery(num_data_blocks, block_bitmap)) return XorecResult::Success;
  if (!is_recoverable(num_data_blocks, num_parity_blocks, block_bitmap)) return XorecResult::DecodeFailure;

  cudaDeviceSynchronize();
  for (unsigned i = 0; i < num_data_blocks; ++i) {
    if (block_bitmap[i]) continue;

    void * XOREC_RESTRICT recover_block = reinterpret_cast<void*>(data_buf + i * block_size);
    const void * XOREC_RESTRICT parity_block = reinterpret_cast<const void*>(parity_buf + (i % num_parity_blocks) * block_size);

    // Copy the parity block to the recover block
    cudaMemcpy(recover_block, parity_block, block_size, cudaMemcpyHostToHost);

    // XOR the recover block with the other data blocks
    for (unsigned j = i % num_parity_blocks; j < num_data_blocks; j += num_parity_blocks) {
      if (i == j) continue;

      const void * XOREC_RESTRICT data_block = reinterpret_cast<const void*>(data_buf + j * block_size);
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
        case XorecVersion::AVX512:
          xorec_xor_blocks_avx512(recover_block, data_block, block_size);
          break;
      }
    }
  }
  return XorecResult::Success;
}






XorecResult xorec_gpu_prefetch_encode(
  const uint8_t *XOREC_RESTRICT gpu_data_buf,
  uint8_t *XOREC_RESTRICT cpu_data_buf,
  uint8_t *XOREC_RESTRICT parity_buf,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  size_t prefetch_bytes,
  XorecVersion version
) {
  cudaMemcpyAsync(cpu_data_buf, gpu_data_buf, num_data_blocks * block_size, cudaMemcpyDeviceToHost);
  if (!XOREC_INIT_CALLED) throw_error("xorec_init() must be called before calling xorec_gpu_prefetch_encode()");
  if (!XOREC_INIT_CALLED) throw_error("xorec_init() must be called before calling xorec_unified_prefetch_encode()");

  XorecResult err = xorec_check_args(cpu_data_buf, parity_buf, block_size, num_data_blocks, num_parity_blocks);
  if (err != XorecResult::Success) return err;

  cudaMemcpyAsync(parity_buf, cpu_data_buf, num_parity_blocks * block_size, cudaMemcpyHostToHost);
  cudaDeviceSynchronize();

  for (unsigned i = num_parity_blocks; i < num_data_blocks; ++i) {
    void * XOREC_RESTRICT parity_block = reinterpret_cast<void*>(parity_buf + (i % num_parity_blocks) * block_size);
    const void * XOREC_RESTRICT data_block = reinterpret_cast<const void*>(cpu_data_buf + i * block_size);

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
      case XorecVersion::AVX512:
        xorec_xor_blocks_avx512(parity_block, data_block, block_size);
        break;
    }
  }  
  return XorecResult::Success;
}


/**
 * @brief Decodes data using XOR-based erasure coding.
 * 
 * @param data_buf Pointer to the data buffer.
 * @param parity_buf Pointer to the parity buffer.
 * @param block_size Size of each block in bytes.
 * @param num_data_blocks Number of data blocks.
 * @param num_parity_blocks Number of parity blocks.
 * @param block_bitmap A bitmap indicating which blocks are present.
 * @return XorecResult XorecResult indicating success or failure.
 */
XorecResult xorec_gpu_prefetch_decode(
  uint8_t *XOREC_RESTRICT gpu_data_buf,
  uint8_t *XOREC_RESTRICT cpu_data_buf,
  const uint8_t *XOREC_RESTRICT parity_buf,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  const uint8_t * XOREC_RESTRICT block_bitmap,
  size_t prefetch_bytes,
  XorecVersion version
) {
  cudaMemcpyAsync(cpu_data_buf, gpu_data_buf, num_data_blocks * block_size, cudaMemcpyDeviceToHost);


  if (!XOREC_INIT_CALLED) throw_error("xorec_init() must be called before calling xorec_unified_prefetch_decode()");

  XorecResult err = xorec_check_args(cpu_data_buf, parity_buf, block_size, num_data_blocks, num_parity_blocks);
  if (err != XorecResult::Success) return err;
  if (!require_recovery(num_data_blocks, block_bitmap)) return XorecResult::Success;
  if (!is_recoverable(num_data_blocks, num_parity_blocks, block_bitmap)) return XorecResult::DecodeFailure;

  cudaDeviceSynchronize();

  for (unsigned i = 0; i < num_data_blocks; ++i) {
    if (block_bitmap[i]) continue;
    void * XOREC_RESTRICT recover_block = reinterpret_cast<void*>(cpu_data_buf + i * block_size);
    const void * XOREC_RESTRICT parity_block = reinterpret_cast<const void*>(parity_buf + (i % num_parity_blocks) * block_size);

    cudaMemcpy(recover_block, parity_block, block_size, cudaMemcpyHostToHost);

    for (unsigned j = i % num_parity_blocks; j < num_data_blocks; j += num_parity_blocks) {
      if (i == j) continue;

      const void * XOREC_RESTRICT data_block = reinterpret_cast<const void*>(cpu_data_buf + j * block_size);
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
        case XorecVersion::AVX512:
          xorec_xor_blocks_avx512(recover_block, data_block, block_size);
          break;
      }
    }
    
    cudaMemcpyAsync(gpu_data_buf + i * block_size, recover_block, block_size, cudaMemcpyHostToDevice);
  }
  return XorecResult::Success;
}