#include "xorec_gpu.cuh"
#include <iostream>
#include "utils.h"



__host__ XorecResult xorec_gpu_encode(
  const uint8_t *XOREC_RESTRICT data_buffer,
  uint8_t *XOREC_RESTRICT parity_buffer,
  uint32_t block_size,
  uint32_t num_data_blocks,
  uint32_t num_parity_blocks
) {
  if (xorec_check_args(block_size, num_data_blocks, num_parity_blocks) != XorecResult::Success) return XorecResult::InvalidCounts;

  cudaMemset(parity_buffer, 0, block_size * num_parity_blocks);
  xorec_gpu_encode_kernel<<<1, MAX_THREADS_PER_BLOCK>>>(data_buffer, parity_buffer, block_size, num_data_blocks, num_parity_blocks);

  return XorecResult::Success;
}


__global__ void xorec_gpu_encode_kernel(
  const uint8_t * XOREC_RESTRICT data_buffer,
  uint8_t * XOREC_RESTRICT parity_buffer,
  uint32_t block_size,
  uint32_t num_data_blocks,
  uint32_t num_parity_blocks
) {
  if (gridDim.x != 1) {
    std::cerr << "Error: xorec_gpu_encode_kernel called with more than one block" << std::endl;
    return;
  }

  uint32_t num_warps = blockDim.x / WARP_SIZE;
  uint32_t warp_idx = threadIdx.x / WARP_SIZE;

  uint32_t block_elems = block_size / sizeof(uint64_t); // number of 64-bit elements in a block
  uint32_t thread_idx = threadIdx.x % WARP_SIZE;

  for (uint32_t i = warp_idx; i < num_data_blocks; i += num_warps) {
    const uint64_t * XOREC_RESTRICT data_block = reinterpret_cast<const uint64_t*>(data_buffer + i * block_size);
    uint64_t * XOREC_RESTRICT parity_block = reinterpret_cast<uint64_t*>(parity_buffer + (i%num_parity_blocks) * block_size);

    for (uint32_t j = thread_idx; j < block_elems; j += WARP_SIZE) {
      parity_block[j] ^= data_block[j];
    }
  }
}
