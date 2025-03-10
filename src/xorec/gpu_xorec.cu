#include "xorec_utils.h"
#include "gpu_xorec.cuh"

namespace XorecGPU {


  XORResult xor_encode(
    const uint8_t * XOR_RESTRICT data_buffer,
    uint8_t * XOR_RESTRICT parity_buffer,
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

    cudaMemset(parity_buffer, 0, num_parity_blocks * block_size);

    int threads_per_block = 128;
    int blocks_per_grid = (num_parity_blocks + threads_per_block - 1) / threads_per_block;
    xor_encode_kernel<<<blocks_per_grid, threads_per_block>>>(data_buffer, parity_buffer, block_size, num_data_blocks, num_parity_blocks);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return XORResult::KernelFailure;
    return XORResult::Success;
  }

  XORResult xor_decode(
    uint8_t * XOR_RESTRICT data_buffer,
    const uint8_t * XOR_RESTRICT parity_buffer,
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
  int threads_per_block = 128;
  int blocks_per_grid = (num_data_blocks + threads_per_block - 1) / threads_per_block;
  xor_decode_kernel<<<blocks_per_grid, threads_per_block>>>(data_buffer, parity_buffer, block_size, num_data_blocks, num_parity_blocks, block_bitmap);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return XORResult::KernelFailure;
  return XORResult::Success;
}




  __global__ void xor_encode_kernel(
    const uint8_t * XOR_RESTRICT data_buffer,
    uint8_t * XOR_RESTRICT parity_buffer,
    uint32_t block_size,
    uint32_t num_data_blocks,
    uint32_t num_parity_blocks
  ) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    for (uint32_t i = idx; i < num_parity_blocks; i += total_threads) {
      void * XOR_RESTRICT parity_block = reinterpret_cast<void*>(parity_buffer + i * block_size);
      for (uint32_t j = i; j < num_data_blocks; j += num_parity_blocks) {
        const void * XOR_RESTRICT data_block = reinterpret_cast<const void*>(data_buffer + j * block_size);
        xor_blocks_kernel(parity_block, data_block, block_size);
      }
    }
  }


  __global__ void xor_decode_kernel(
    uint8_t * XOR_RESTRICT data_buffer,
    const uint8_t * XOR_RESTRICT parity_buffer,
    uint32_t block_size,
    uint32_t num_data_blocks,
    uint32_t num_parity_blocks,
    std::bitset<256> block_bitmap
  ) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    for (uint32_t i = idx; i < num_data_blocks; i += total_threads) {
      if (block_bitmap.test(i)) continue;

      void * XOR_RESTRICT recover_block = reinterpret_cast<void*>(data_buffer + i * block_size);

      const void * XOR_RESTRICT parity_block = reinterpret_cast<const void*>(parity_buffer + (i % num_parity_blocks) * block_size);

      cudaMemcpy(recover_block, parity_block, block_size, cudaMemcpyDeviceToDevice);
      for (uint32_t j = i % num_parity_blocks; j < num_data_blocks; j += num_parity_blocks) {
        if (i == j) continue;
        const void * XOR_RESTRICT data_block = reinterpret_cast<const void*>(data_buffer + j * block_size);
        xor_blocks_kernel(recover_block, data_block, block_size);
      }
    }
  }

}
