#ifndef XOREC_GPU_KERNELS_CUH
#define XOREC_GPU_KERNELS_CUH

#include <cuda_runtime.h>
#include "xorec_utils.h"


__host__ XORResult xorec_gpu_encode(
  const uint8_t *XOR_RESTRICT data_buffer,
  uint8_t *XOR_RESTRICT parity_buffer,
  uint32_t block_size,
  uint32_t num_data_blocks,
  uint32_t num_parity_blocks
);

__global__ void xorec_gpu_encode_kernel(
  const uint8_t * XOR_RESTRICT data_buffer,
  uint8_t * XOR_RESTRICT parity_buffer,
  uint32_t block_size,
  uint32_t num_data_blocks,
  uint32_t num_parity_blocks
);


#endif // XOREC_GPU_KERNELS_CUH