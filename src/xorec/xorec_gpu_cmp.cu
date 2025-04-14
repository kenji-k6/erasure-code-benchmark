#include "xorec_gpu_cmp.cuh"
#include "utils.hpp"

int DEVICE_ID;
int NUM_BLOCKS;
int THREADS_PER_BLOCK;
size_t TOT_THREADS;
size_t ATOMIC_XOR_ELEMS_BLOCK;
size_t ATOMIC_XOR_ELEMS_TOT;

static bool XOREC_GPU_INIT_CALLED = false;

void xorec_gpu_init(int num_gpu_blocks, int threads_per_block, size_t num_data_blocks, size_t block_size) {
  if (XOREC_GPU_INIT_CALLED) return;

  int device_count;

  cudaGetDeviceCount(&device_count);

  if (device_count <= 0) throw_error("No CUDA devices found");

  DEVICE_ID = 0;
  cudaError_t err = cudaSetDevice(DEVICE_ID);
  if (err != cudaSuccess) throw_error("Failed to set device");
  
  cudaDeviceProp device_prop;
  err = cudaGetDeviceProperties(&device_prop, DEVICE_ID);
  if (err != cudaSuccess) throw_error("Failed to rertrieve device properties");

  COMPLETE_DATA_BITMAP.resize(num_data_blocks);
  std::fill_n(COMPLETE_DATA_BITMAP.begin(), num_data_blocks, 1);

  NUM_BLOCKS = num_gpu_blocks;
  THREADS_PER_BLOCK = threads_per_block;

  
  
  if (num_gpu_blocks <= 0 || threads_per_block <= 0) throw_error("Invalid block count or threads per block");
  if (threads_per_block > device_prop.maxThreadsPerBlock) throw_error("Threads per block exceeds device limit");

  TOT_THREADS = NUM_BLOCKS * THREADS_PER_BLOCK;
  ATOMIC_XOR_ELEMS_BLOCK = block_size / sizeof(CUDA_ATOMIC_XOR_T);
  ATOMIC_XOR_ELEMS_TOT = num_data_blocks * ATOMIC_XOR_ELEMS_BLOCK;

  XOREC_GPU_INIT_CALLED = true;
}

XorecResult xorec_gpu_encode(
  const uint8_t *XOREC_RESTRICT data_buf,
  uint8_t *XOREC_RESTRICT parity_buf,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks
) {
  if (!XOREC_GPU_INIT_CALLED) throw_error("xorec_gpu_init() must be called before calling xorec_encode()");
  XorecResult err = xorec_check_args(data_buf, parity_buf, block_size, num_data_blocks, num_parity_blocks);
  if (err != XorecResult::Success) return err;
  if (block_size % sizeof(CUDA_ATOMIC_XOR_T) != 0) return XorecResult::InvalidSize;

  cudaMemsetAsync(parity_buf, 0, block_size * num_parity_blocks);
  xorec_gpu_xor_parity_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(data_buf, parity_buf, block_size, num_data_blocks, num_parity_blocks, TOT_THREADS, ATOMIC_XOR_ELEMS_BLOCK, ATOMIC_XOR_ELEMS_TOT);
  return XorecResult::Success;
}


XorecResult xorec_gpu_decode(
  uint8_t *XOREC_RESTRICT data_buf,
  uint8_t *XOREC_RESTRICT parity_buf,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  const uint8_t *XOREC_RESTRICT block_bitmap   ///< Indexing for parity blocks starts at bit 128, e.g. the j-th parity block is at bit 128 + j, j < 128
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

  xorec_gpu_xor_parity_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(data_buf, parity_buf, block_size, num_data_blocks, num_parity_blocks, TOT_THREADS, ATOMIC_XOR_ELEMS_BLOCK, ATOMIC_XOR_ELEMS_TOT);

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
  size_t num_parity_blocks,
  size_t tot_threads,
  size_t atomic_xor_elems_block,
  size_t atomic_xor_elems_tot
) {
  unsigned t_idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (unsigned i = t_idx; i < atomic_xor_elems_tot; i += tot_threads) {
    unsigned block_idx = i / atomic_xor_elems_block;
    unsigned parity_idx = block_idx % num_parity_blocks;

    const CUDA_ATOMIC_XOR_T * XOREC_RESTRICT data_block_64 = reinterpret_cast<const CUDA_ATOMIC_XOR_T*>(data_buf + block_idx * block_size);
    CUDA_ATOMIC_XOR_T * XOREC_RESTRICT parity_block_64 = reinterpret_cast<CUDA_ATOMIC_XOR_T*>(parity_buf + parity_idx * block_size);

    atomicXor(&parity_block_64[i % atomic_xor_elems_block], data_block_64[i % atomic_xor_elems_block]);
  }
}

