#ifndef XOREC_GPU_CMP_CUH
#define XOREC_GPU_CMP_CUH

#include <cuda_runtime.h>
#include "xorec_utils.hpp"

#define CUDA_ATOMIC_XOR_T unsigned long long int

void xorec_gpu_init();

XorecResult xorec_gpu_encode(
  const uint8_t *XOREC_RESTRICT data_buffer,
  uint8_t *XOREC_RESTRICT parity_buffer,
  uint32_t block_size,
  uint32_t num_data_blocks,
  uint32_t num_parity_blocks
);

XorecResult xorec_gpu_decode(
  uint8_t *XOREC_RESTRICT data_buffer,
  const uint8_t *XOREC_RESTRICT parity_buffer,
  uint32_t block_size,
  uint32_t num_data_blocks,
  uint32_t num_parity_blocks,
  const uint8_t *block_bitmap   ///< Indexing for parity blocks starts at bit 128, e.g. the j-th parity block is at bit 128 + j, j < 128
);

__global__ void xorec_gpu_encode_kernel(
  const uint8_t * XOREC_RESTRICT data_buffer,
  uint8_t * XOREC_RESTRICT parity_buffer,
  uint32_t block_size,
  uint32_t num_data_blocks,
  uint32_t num_parity_blocks
);


__global__ void xorec_gpu_decode_kernel(
  uint8_t * XOREC_RESTRICT recover_block,
  const uint8_t *XOREC_RESTRICT data_buffer,
  uint32_t block_size,
  uint32_t num_data_blocks,
  uint32_t num_parity_blocks,
  uint32_t lost_block_idx,
  uint32_t parity_idx
);



#endif // XOREC_GPU_CMP_CUH