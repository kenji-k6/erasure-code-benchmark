#ifndef XOREC_GPU_CMP_CUH
#define XOREC_GPU_CMP_CUH

#include <cuda_runtime.h>
#include "xorec_utils.hpp"

#define CUDA_ATOMIC_XOR_T unsigned long long int

/**
 * @brief Initialize the necessary global variables & GPU environment for XOR encoding and decoding on the GPU.
 */
void xorec_gpu_init();


/**
 * @brief Runs the XOR encoding algorithm on the GPU.
 * 
 * @param data_buffer The data buffer to encode. Must be allocated in unified memory.
 * @param parity_buffer The parity buffer to write the encoded parity blocks to. Must be allocated in unified memory.
 * @param block_size Size of each block in bytes.
 * @param num_data_blocks Number of data blocks.
 * @param num_parity_blocks Number of parity blocks.
 * @return XorecResult 
 */
XorecResult xorec_gpu_encode(
  const uint8_t *XOREC_RESTRICT data_buffer,
  uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks
);

/**
 * @brief Runs the XOR decoding algorithm on the GPU.
 * 
 * @param data_buffer The data buffer to encode. Must be allocated in unified memory.
 * @param parity_buffer The parity buffer to write the encoded parity blocks to. Must be allocated in unified memory.
 * @param block_size Size of each block in bytes.
 * @param num_data_blocks Number of data blocks.
 * @param num_parity_blocks Number of parity blocks.
 * @param block_bitmap Bitmap of which blocks are present. Indexing for parity blocks starts at bit 128, e.g. the j-th parity block is at bit 128 + j, j < 128
 * @return XorecResult 
 */
XorecResult xorec_gpu_decode(
  uint8_t *XOREC_RESTRICT data_buffer,
  const uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  const uint8_t *block_bitmap
);

/**
 * @brief CUDA kernel for XOR encoding.
 * 
 * @param data_buffer 
 * @param parity_buffer 
 * @param block_size 
 * @param num_data_blocks 
 * @param num_parity_blocks 
 * @return __global__ 
 */
__global__ void xorec_gpu_encode_kernel(
  const uint8_t * XOREC_RESTRICT data_buffer,
  uint8_t * XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks
);


/**
 * @brief CUDA kernel for XOR decoding.
 * 
 * @param recover_block 
 * @param data_buffer 
 * @param block_size 
 * @param num_data_blocks 
 * @param num_parity_blocks 
 * @param lost_block_idx 
 * @param parity_idx 
 * @return __global__ 
 */
__global__ void xorec_gpu_decode_kernel(
  uint8_t * XOREC_RESTRICT recover_block,
  const uint8_t *XOREC_RESTRICT data_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  uint32_t lost_block_idx,
  uint32_t parity_idx
);



#endif // XOREC_GPU_CMP_CUH