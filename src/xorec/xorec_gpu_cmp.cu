#include "xorec_gpu_cmp.cuh"
#include "utils.hpp"

static bool XOREC_GPU_INIT_CALLED = false;

void xorec_gpu_init(size_t num_data_blocks, int device_id) {
  if (XOREC_GPU_INIT_CALLED) return;

  int device_count;

  cudaGetDeviceCount(&device_count);

  if (device_count <= 0) throw_error("No CUDA devices found");

  cudaError_t err = cudaSetDevice(device_id);
  if (err != cudaSuccess) throw_error("Failed to set device");
  
  cudaDeviceProp device_prop;
  err = cudaGetDeviceProperties(&device_prop, device_id);
  if (err != cudaSuccess) throw_error("Failed to rertrieve device properties");

  COMPLETE_DATA_BITMAP.resize(num_data_blocks);
  std::fill_n(COMPLETE_DATA_BITMAP.begin(), num_data_blocks, 1);

  XOREC_GPU_INIT_CALLED = true;
}

XorecResult xorec_gpu_encode(
  const uint8_t *XOREC_RESTRICT data_buf,
  uint8_t *XOREC_RESTRICT parity_buf,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  size_t num_gpu_blocks,
  size_t threads_per_block
) {
  if (!XOREC_GPU_INIT_CALLED) throw_error("xorec_gpu_init() must be called before calling xorec_encode()");
  XorecResult err = xorec_check_args(data_buf, parity_buf, block_size, num_data_blocks, num_parity_blocks);
  if (err != XorecResult::Success) return err;
  if (block_size % sizeof(CUDA_ATOMIC_XOR_T) != 0) return XorecResult::InvalidSize;

  cudaMemsetAsync(parity_buf, 0, block_size * num_parity_blocks);
  xorec_gpu_xor_parity_kernel<<<num_gpu_blocks, threads_per_block>>>(data_buf, parity_buf, block_size, num_data_blocks, num_parity_blocks);
  return XorecResult::Success;
}


XorecResult xorec_gpu_decode(
  uint8_t *XOREC_RESTRICT data_buf,
  uint8_t *XOREC_RESTRICT parity_buf,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  const uint8_t *XOREC_RESTRICT block_bitmap,   ///< Indexing for parity blocks starts at bit 128, e.g. the j-th parity block is at bit 128 + j, j < 128
  size_t num_gpu_blocks,
  size_t threads_per_block
) {
  if (!XOREC_GPU_INIT_CALLED) throw_error("xorec_gpu_init() must be called before calling xorec_encode()");

  XorecResult err = xorec_check_args(data_buf, parity_buf, block_size, num_data_blocks, num_parity_blocks);
  if (err != XorecResult::Success) return err;
  if (block_size % sizeof(CUDA_ATOMIC_XOR_T) != 0) return XorecResult::InvalidSize;

  if (!require_recovery(num_data_blocks, block_bitmap)) return XorecResult::Success;
  if (!is_recoverable(num_data_blocks, num_parity_blocks, block_bitmap)) return XorecResult::DecodeFailure;

  
  // Zero out lost blocks
  for (uint32_t i = 0; i < num_data_blocks; ++i) {
    if (!block_bitmap[i]) cudaMemsetAsync(data_buf + i * block_size, 0, block_size);
  }

  xorec_gpu_xor_parity_kernel<<<num_gpu_blocks, threads_per_block>>>(data_buf, parity_buf, block_size, num_data_blocks, num_parity_blocks);

  for (uint32_t i = 0; i < num_data_blocks; ++i) {
    if (!block_bitmap[i]) cudaMemcpyAsync(data_buf + i * block_size, parity_buf + (i % num_parity_blocks) * block_size, block_size, cudaMemcpyDeviceToDevice);
  }
  return XorecResult::Success;
}

__global__ void xorec_gpu_xor_parity_kernel(
  const uint8_t * XOREC_RESTRICT data_buf,
  uint8_t * XOREC_RESTRICT parity_buf,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks
) {
  unsigned num_threads = blockDim.x * gridDim.x;
  unsigned glbl_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned block_elems = block_size / sizeof(CUDA_ATOMIC_XOR_T);
  unsigned tot_elems = num_data_blocks * block_elems;

  for (unsigned i = glbl_thread_idx; i < tot_elems; i += num_threads) {
    unsigned block_idx = i / block_elems;
    unsigned parity_idx = block_idx % num_parity_blocks;

    const CUDA_ATOMIC_XOR_T * XOREC_RESTRICT data_block_64 = reinterpret_cast<const CUDA_ATOMIC_XOR_T*>(data_buf + block_idx * block_size);
    CUDA_ATOMIC_XOR_T * XOREC_RESTRICT parity_block_64 = reinterpret_cast<CUDA_ATOMIC_XOR_T*>(parity_buf + parity_idx * block_size);

    atomicXor(&parity_block_64[i % block_elems], data_block_64[i % block_elems]);
  }
}

