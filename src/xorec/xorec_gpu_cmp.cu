#include "xorec_gpu_cmp.cuh"
#include "utils.hpp"

int DEVICE_ID;
int MAX_THREADS_PER_BLOCK; 

__device__ __constant__ int WARP_SIZE;

static bool XOREC_GPU_INIT_CALLED = false;

void xorec_gpu_init() {
  if (XOREC_GPU_INIT_CALLED) return;
  XOREC_GPU_INIT_CALLED = true;

  int device_count;

  cudaGetDeviceCount(&device_count);

  if (device_count <= 0) throw_error("No CUDA devices found");

  DEVICE_ID = 0;
  cudaError_t err = cudaSetDevice(DEVICE_ID);
  if (err != cudaSuccess) throw_error("Failed to set device");
  
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, DEVICE_ID);
  MAX_THREADS_PER_BLOCK = device_prop.maxThreadsPerBlock;
  int warp_size = device_prop.warpSize;

  err = cudaMemcpyToSymbol(WARP_SIZE, &warp_size, sizeof(int));
  if (err != cudaSuccess) throw_error("Failed to copy warp size to constant memory"); 

  std::fill_n(COMPLETE_DATA_BITMAP.begin(), XOREC_MAX_DATA_BLOCKS, 1);
}

XorecResult xorec_gpu_encode(
  const uint8_t *XOREC_RESTRICT data_buffer,
  uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks
) {
  if (xorec_check_args(block_size, num_data_blocks, num_parity_blocks) != XorecResult::Success) return XorecResult::InvalidCounts;
  if (block_size % sizeof(CUDA_ATOMIC_XOR_T) != 0) return XorecResult::InvalidSize;

  cudaMemset(parity_buffer, 0, block_size * num_parity_blocks);
  xorec_gpu_encode_kernel<<<1, MAX_THREADS_PER_BLOCK>>>(data_buffer, parity_buffer, block_size, num_data_blocks, num_parity_blocks);

  return XorecResult::Success;
}


XorecResult xorec_gpu_decode(
  uint8_t *XOREC_RESTRICT data_buffer,
  const uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  const uint8_t *XOREC_RESTRICT block_bitmap   ///< Indexing for parity blocks starts at bit 128, e.g. the j-th parity block is at bit 128 + j, j < 128
) {
  if (!recovery_needed(block_bitmap)) return XorecResult::Success;

  if (xorec_check_args(block_size, num_data_blocks, num_parity_blocks) != XorecResult::Success) return XorecResult::InvalidCounts;
  if (block_size % sizeof(CUDA_ATOMIC_XOR_T) != 0) return XorecResult::InvalidSize;

  if (!is_recoverable(block_bitmap, num_data_blocks, num_parity_blocks)) return XorecResult::DecodeFailure;
  for (uint32_t i = 0; i < num_data_blocks;  ++i) {
    if (block_bitmap[i]) continue;
    uint8_t * XOREC_RESTRICT recover_block = data_buffer + i * block_size;
    const uint8_t * XOREC_RESTRICT parity_block = parity_buffer + (i % num_parity_blocks) * block_size;
    cudaMemcpy(recover_block, parity_block, block_size, cudaMemcpyDeviceToDevice);
    xorec_gpu_decode_kernel<<<1, MAX_THREADS_PER_BLOCK>>>(recover_block, data_buffer, block_size, num_data_blocks, num_parity_blocks, i, i%num_parity_blocks);
  }

  return XorecResult::Success;
}

__global__ void xorec_gpu_decode_kernel(
  uint8_t * XOREC_RESTRICT recover_block,
  const uint8_t *XOREC_RESTRICT data_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  uint32_t lost_block_idx,
  uint32_t parity_idx
) {
  unsigned num_warps = blockDim.x / WARP_SIZE;
  unsigned warp_idx = threadIdx.x / WARP_SIZE;

  unsigned block_elems = block_size / sizeof(CUDA_ATOMIC_XOR_T); // number of 64-bit elements in a block
  unsigned thread_idx = threadIdx.x % WARP_SIZE;
  
  CUDA_ATOMIC_XOR_T * XOREC_RESTRICT recover_block_64 = reinterpret_cast<CUDA_ATOMIC_XOR_T*>(recover_block);
  for (unsigned i = parity_idx + warp_idx * num_parity_blocks; i < num_data_blocks; i += num_warps * num_parity_blocks) {
    if (i == lost_block_idx) continue;
    const CUDA_ATOMIC_XOR_T * XOREC_RESTRICT data_block_64 = reinterpret_cast<const CUDA_ATOMIC_XOR_T*>(data_buffer + i * block_size);

    for (uint32_t j = thread_idx; j < block_elems; j += WARP_SIZE) {
      atomicXor(&recover_block_64[j], data_block_64[j]);
    }
  }
}



__global__ void xorec_gpu_encode_kernel(
  const uint8_t * XOREC_RESTRICT data_buffer,
  uint8_t * XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks
) {
  unsigned num_warps = blockDim.x / WARP_SIZE;
  unsigned warp_idx = threadIdx.x / WARP_SIZE;

  unsigned block_elems = block_size / sizeof(CUDA_ATOMIC_XOR_T); // number of 64-bit elements in a block
  unsigned thread_idx = threadIdx.x % WARP_SIZE;

  for (unsigned i = warp_idx; i < num_data_blocks; i += num_warps) {
    const CUDA_ATOMIC_XOR_T * XOREC_RESTRICT data_block_64 = reinterpret_cast<const CUDA_ATOMIC_XOR_T*>(data_buffer + i * block_size);
    CUDA_ATOMIC_XOR_T * XOREC_RESTRICT parity_block_64 = reinterpret_cast<CUDA_ATOMIC_XOR_T*>(parity_buffer + (i%num_parity_blocks) * block_size);

    for (unsigned j = thread_idx; j < block_elems; j += WARP_SIZE) {
      atomicXor(&parity_block_64[j], data_block_64[j]);
    }
  }
}

