#include "xorec_gpu_cmp.cuh"
#include <iostream>
#include "utils.hpp"

int DEVICE_ID;
int MAX_THREADS_PER_BLOCK; 

__constant__ int WARP_SIZE;

static bool XOREC_GPU_INIT_CALLED = false;

void xorec_gpu_init() {
  if (XOREC_GPU_INIT_CALLED) return;
  XOREC_GPU_INIT_CALLED = true;

  int device_count = -1;

  cudaGetDeviceCount(&device_count);

  if (device_count <= 0) throw_error("No CUDA devices found");

  DEVICE_ID = 0;
  
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, DEVICE_ID);
  MAX_THREADS_PER_BLOCK = device_prop.maxThreadsPerBlock;
  int warp_size = device_prop.warpSize;

  std::fill_n(COMPLETE_DATA_BITMAP.begin(), XOREC_MAX_DATA_BLOCKS, 1);

  cudaMemcpyToSymbol(WARP_SIZE, &warp_size, sizeof(int));
  cudaSetDevice(DEVICE_ID);
}

XorecResult xorec_gpu_encode(
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


XorecResult xorec_gpu_decode(
  uint8_t *XOREC_RESTRICT data_buffer,
  const uint8_t *XOREC_RESTRICT parity_buffer,
  uint32_t block_size,
  uint32_t num_data_blocks,
  uint32_t num_parity_blocks,
  const uint8_t *XOREC_RESTRICT block_bitmap   ///< Indexing for parity blocks starts at bit 128, e.g. the j-th parity block is at bit 128 + j, j < 128
) {
  if (!recovery_needed(block_bitmap)) return XorecResult::Success;

  if (xorec_check_args(block_size, num_data_blocks, num_parity_blocks) != XorecResult::Success) return XorecResult::InvalidCounts;

  if (!is_recoverable(block_bitmap, num_data_blocks, num_parity_blocks)) return XorecResult::DecodeFailure;

  for (uint32_t i = 0; i < num_data_blocks;  ++i) {
    if (block_bitmap[i]) continue;
    uint8_t * XOREC_RESTRICT recover_block = data_buffer + i * block_size;
    const uint8_t * XOREC_RESTRICT parity_block = parity_buffer + (i % num_parity_blocks) * block_size;

    cudaMemcpy(recover_block, parity_block, block_size, cudaMemcpyDeviceToDevice);
    xorec_gpu_decode_kernel<<<1, MAX_THREADS_PER_BLOCK>>>(data_buffer, parity_buffer, block_size, num_data_blocks, num_parity_blocks, i, i%num_parity_blocks);
  }

  return XorecResult::Success;
}

__global__ void xorec_gpu_decode_kernel(
  uint8_t * XOREC_RESTRICT recover_block,
  const uint8_t *XOREC_RESTRICT data_buffer,
  uint32_t block_size,
  uint32_t num_data_blocks,
  uint32_t num_parity_blocks,
  uint32_t lost_block_idx,
  uint32_t parity_idx
) {
  uint32_t num_warps = blockDim.x / WARP_SIZE;
  uint32_t warp_idx = threadIdx.x / WARP_SIZE;

  uint32_t block_elems = block_size / sizeof(uint64_t); // number of 64-bit elements in a block
  uint32_t thread_idx = threadIdx.x % WARP_SIZE;
  
  // pidx, pidx + warpidx*num_parity_blocks
  uint64_t * XOREC_RESTRICT recover_block_64 = reinterpret_cast<uint64_t*>(recover_block);
  for (uint32_t i = parity_idx + warp_idx * num_parity_blocks; i < num_data_blocks; i += num_warps * num_parity_blocks) {
    if (i == lost_block_idx) continue;
    const uint64_t * XOREC_RESTRICT data_block_64 = reinterpret_cast<const uint64_t*>(data_buffer + i * block_size);

    for (uint32_t j = thread_idx; j < block_elems; j += WARP_SIZE) {
      recover_block_64[j] ^= data_block_64[j];
    }
  }
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
    const uint64_t * XOREC_RESTRICT data_block_64 = reinterpret_cast<const uint64_t*>(data_buffer + i * block_size);
    uint64_t * XOREC_RESTRICT parity_block_64 = reinterpret_cast<uint64_t*>(parity_buffer + (i%num_parity_blocks) * block_size);

    for (uint32_t j = thread_idx; j < block_elems; j += WARP_SIZE) {
      parity_block_64[j] ^= data_block_64[j];
    }
  }
}

