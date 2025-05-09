#ifndef XOREC_GPU_CMP_CUH
#define XOREC_GPU_CMP_CUH

#include <cuda_runtime.h>
#include "xorec_utils.hpp"

#define CUDA_ATOMIC_T unsigned long long int

extern std::vector<uint8_t> COMPLETE_DATA_BITMAP;

/**
 * @brief Initialize the necessary global variables & GPU environment for XOR encoding and decoding on the GPU.
 */
void xorec_gpu_init(size_t num_data_blocks, int device_id=0);

/**
 * @brief Runs the XOR encoding algorithm on the GPU.
 * 
 * @param data_buf The data buffer to encode. Must be allocated in unified memory.
 * @param parity_buf The parity buffer to write the encoded parity blocks to. Must be allocated in unified memory.
 * @param num_chunks Number of chunks
 * @param block_size Size of each block in bytes.
 * @param chunk_data_blocks Number of data blocks per chunk.
 * @param chunk_parity_blocks Number of parity blocks per chunk.
 * @param num_gpu_blocks Number of GPU blocks to use for encoding.
 * @param threads_per_block Number of threads per block to use for encoding.
 * @return XorecResult 
 */
XorecResult xorec_gpu_encode(
  const uint8_t *XOREC_RESTRICT data_buf,
  uint8_t *XOREC_RESTRICT parity_buf,
  size_t num_chunks,
  size_t block_size,
  size_t chunk_data_blocks,
  size_t chunk_parity_blocks,
  size_t num_gpu_blocks,
  size_t threads_per_block
);

/**
 * @brief Runs the XOR decoding algorithm on the GPU.
 * 
 * @attention LOST BLOCKS MUST BE ZEROED OUT BEFOREHAND
 * 
 * @param data_buf The data buffer to recover. Must be allocated in unified memory.
 * @param parity_buf The parity buffer to write the encoded parity blocks to. Must be allocated in unified memory.
 * @param num_chunks Number of chunks
 * @param block_size Size of each block in bytes.
 * @param chunk_data_blocks Number of data blocks per chunk.
 * @param chunk_parity_blocks Number of parity blocks per chunk.
 * @param num_gpu_blocks Number of GPU blocks to use for encoding.
 * @param threads_per_block Number of threads per block to use for encoding.
 * @param block_bitmap Bitmap of which blocks are present. Indexing for parity blocks starts at bit 128, e.g. the j-th parity block is at bit 128 + j, j < 128
 * @param device_block_bitmap Pointer to device memory region of the same size as the block bitmap. This is used to store the block bitmap on the device.
 * @param num_gpu_blocks Number of GPU blocks to use for decoding.
 * @param threads_per_block Number of threads per block to use for decoding.
 * @return XorecResult 
 */
XorecResult xorec_gpu_decode(
  uint8_t* XOREC_RESTRICT data_buf,
  uint8_t* XOREC_RESTRICT parity_buf,
  size_t num_chunks,
  size_t block_size,
  size_t chunk_data_blocks,
  size_t chunk_parity_blocks,
  const uint8_t* XOREC_RESTRICT block_bitmap,
  uint8_t* XOREC_RESTRICT device_block_bitmap,
  size_t num_gpu_blocks,
  size_t threads_per_block
);


/**
 * @brief XOR-EC GPU XOR kernel.
 */
__global__ void xorec_gpu_xor_kernel(
  const uint8_t* XOREC_RESTRICT data_buf,
  uint8_t* XOREC_RESTRICT parity_buf,
  size_t num_chunks,
  size_t block_size,
  size_t chunk_data_blocks,
  size_t chunk_parity_blocks
);

/**
 * @brief XOR-EC GPU zero kernel.
 * 
 * @param data_buf data buffer
 * @param num_chunks Number of chunks 
 * @param block_size Size of each block in bytes.
 * @param chunk_data_blocks Number of data blocks per chunk.
 * @param chunk_parity_blocks Number of parity blocks per chunk.
 * @param block_bitmap Bitmap of which blocks are present. Allocated in GPU memory.
 */
__global__ void xorec_gpu_zero_kernel(
  uint8_t* XOREC_RESTRICT data_buf,
  size_t num_chunks,
  size_t block_size,
  size_t chunk_data_blocks,
  size_t chunk_parity_blocks,
  const uint8_t* block_bitmap
);

/**
 * @brief XOR-EC GPU recovery kernel. Used during decoding.
 * 
 * @param data_buf data buffer
 * @param parity_buf parity buffer
 * @param num_chunks Number of chunks 
 * @param block_size Size of each block in bytes.
 * @param chunk_data_blocks Number of data blocks per chunk.
 * @param chunk_parity_blocks Number of parity blocks per chunk.
 * @param block_bitmap Bitmap of which blocks are present. Allocated in GPU memory.
 */
__global__ void xorec_gpu_recover_kernel(
  uint8_t* XOREC_RESTRICT data_buf,
  const uint8_t* XOREC_RESTRICT parity_buf,
  size_t num_chunks,
  size_t block_size,
  size_t chunk_data_blocks,
  size_t chunk_parity_blocks,
  const uint8_t* block_bitmap
);


#endif // XOREC_GPU_CMP_CUH