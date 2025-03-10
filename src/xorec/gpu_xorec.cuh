#ifndef GPU_XOREC_CUH
#define GPU_XOREC_CUH

#include "xorec_utils.h"
#include <cuda_runtime.h>

namespace XorecGPU {
  __global__ void xor_encode_kernel(
    const uint8_t * XOR_RESTRICT data_buffer,
    uint8_t * XOR_RESTRICT parity_buffer,
    uint32_t block_size,
    uint32_t num_data_blocks,
    uint32_t num_parity_blocks
  );
  
  __global__ void xor_decode_kernel(
    uint8_t * XOR_RESTRICT data_buffer,
    const uint8_t * XOR_RESTRICT parity_buffer,
    uint32_t block_size,
    uint32_t num_data_blocks,
    uint32_t num_parity_blocks,
    std::bitset<256> block_bitmap
  );
  

  XORResult xor_encode(
    const uint8_t * XOR_RESTRICT data_buffer,
    uint8_t * XOR_RESTRICT parity_buffer,
    uint32_t block_size,
    uint32_t num_data_blocks,
    uint32_t num_parity_blocks
  );

  XORResult xor_decode(
    uint8_t * XOR_RESTRICT data_buffer,
    const uint8_t * XOR_RESTRICT parity_buffer,
    uint32_t block_size,
    uint32_t num_data_blocks,
    uint32_t num_parity_blocks,
    std::bitset<256> block_bitmap
  );
  
  __device__ static void inline xor_blocks_kernel(
    void * XOR_RESTRICT dst,
    const void * XOR_RESTRICT src,
    const uint32_t bytes
  ) {
    uint64_t * XOR_RESTRICT dst64 = reinterpret_cast<uint64_t*>(dst);
    const uint64_t * XOR_RESTRICT src64 = reinterpret_cast<const uint64_t*>(src);
  
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;
  
    // Process in chunks of 4 * 8 bytes (32 bytes per thread)
    for (uint32_t i = idx * 4; i < bytes / 8; i += total_threads * 4) {
      dst64[i] ^= src64[i];
      dst64[i + 1] ^= src64[i + 1];
      dst64[i + 2] ^= src64[i + 2];
      dst64[i + 3] ^= src64[i + 3];
    }
  }
}


#endif // GPU_XOREC_CUH