#include "xorec_gpu_cmp.cuh"
#include "utils.hpp"

int DEVICE_ID;
int NUM_BLOCKS;
int THREADS_PER_BLOCK;

__device__ __constant__ int WARP_SIZE;

static bool XOREC_GPU_INIT_CALLED = false;

void xorec_gpu_init(int num_gpu_blocks, int threads_per_block, size_t num_data_blocks, [[maybe_unused]] size_t num_parity_blocks) {
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
  int warp_size = device_prop.warpSize;

  err = cudaMemcpyToSymbol(WARP_SIZE, &warp_size, sizeof(int));
  if (err != cudaSuccess) throw_error("Failed to copy warp size to constant memory"); 

  std::fill_n(COMPLETE_DATA_BITMAP.begin(), num_data_blocks, 1);

  NUM_BLOCKS = num_gpu_blocks;
  THREADS_PER_BLOCK = threads_per_block;

  if (num_gpu_blocks <= 0 || threads_per_block <= 0) throw_error("Invalid block count or threads per block");
  if (threads_per_block > device_prop.maxThreadsPerBlock) throw_error("Threads per block exceeds device limit");
}

XorecResult xorec_gpu_encode(
  const uint8_t *XOREC_RESTRICT data_buffer,
  uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks
) {
  if (!XOREC_GPU_INIT_CALLED) throw_error("xorec_init() must be called before calling xorec_encode()");
  if (xorec_check_args(block_size, num_data_blocks, num_parity_blocks) != XorecResult::Success) return XorecResult::InvalidCounts;
  if (block_size % sizeof(CUDA_ATOMIC_XOR_T) != 0) return XorecResult::InvalidSize;

  cudaMemsetAsync(parity_buffer, 0, block_size * num_parity_blocks);
  xorec_gpu_xor_parity_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(data_buffer, parity_buffer, block_size, num_data_blocks, num_parity_blocks);

  return XorecResult::Success;
}


XorecResult xorec_gpu_decode(
  uint8_t *XOREC_RESTRICT data_buffer,
  uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  const uint8_t *XOREC_RESTRICT block_bitmap   ///< Indexing for parity blocks starts at bit 128, e.g. the j-th parity block is at bit 128 + j, j < 128
) {
  if (!XOREC_GPU_INIT_CALLED) throw_error("xorec_init() must be called before calling xorec_decode()");
  if (!recovery_needed(block_bitmap)) return XorecResult::Success;

  if (xorec_check_args(block_size, num_data_blocks, num_parity_blocks) != XorecResult::Success) return XorecResult::InvalidCounts;
  if (block_size % sizeof(CUDA_ATOMIC_XOR_T) != 0) return XorecResult::InvalidSize;

  if (!is_recoverable(block_bitmap, num_data_blocks, num_parity_blocks)) return XorecResult::DecodeFailure;

  
  // Zerou out lost blocks
  for (uint32_t i = 0; i < num_data_blocks; ++i) {
    if (!block_bitmap[i]) cudaMemsetAsync(data_buffer + i * block_size, 0, block_size);
  }

  xorec_gpu_xor_parity_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(data_buffer, parity_buffer, block_size, num_data_blocks, num_parity_blocks);

  for (uint32_t i = 0; i < num_data_blocks; ++i) {
    if (!block_bitmap[i]) cudaMemcpyAsync(data_buffer + i * block_size, parity_buffer + (i % num_parity_blocks) * block_size, block_size, cudaMemcpyDeviceToDevice);
  }
  return XorecResult::Success;
}

__global__ void xorec_gpu_xor_parity_kernel(
  const uint8_t * XOREC_RESTRICT data_buffer,
  uint8_t * XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks
) {

  unsigned num_threads = blockDim.x * gridDim.x;
  unsigned glbl_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned block_elems = block_size / sizeof(CUDA_ATOMIC_XOR_T); // number of 64-bit elements in a block
  unsigned tot_elems = block_elems * num_data_blocks;


  for (unsigned i = glbl_thread_idx; i < tot_elems; i += num_threads) {
    unsigned block_idx = i / block_elems;
    unsigned parity_idx = block_idx % num_parity_blocks;

    const CUDA_ATOMIC_XOR_T * XOREC_RESTRICT data_block_64 = reinterpret_cast<const CUDA_ATOMIC_XOR_T*>(data_buffer + block_idx * block_size);
    CUDA_ATOMIC_XOR_T * XOREC_RESTRICT parity_block_64 = reinterpret_cast<CUDA_ATOMIC_XOR_T*>(parity_buffer + parity_idx * block_size);

    atomicXor(&parity_block_64[i % block_elems], data_block_64[i % block_elems]);
  }
}


/**
  * @brief THE FUNCTIONS BELOW SHOULD ONLY BE USED FOR THE BENCHMARKS REQUIRED FOR THE PAPER
  * 
  * THEY ARE MEANT TO MEASURE THE THEORETICAL MAXIMUM THROUGHPUT FOR GPU OPTIMIZED XOR-EC VERSIONS
  */
 
  XorecResult xorec_gpu_encode_full_message(
    const uint8_t *XOREC_RESTRICT data_buffer,
    uint8_t *XOREC_RESTRICT parity_buffer,
    size_t num_chunks,
    size_t block_size,
    size_t data_blks_per_chunk,
    size_t parity_blks_per_chunk
  ) {
    if (!XOREC_GPU_INIT_CALLED) throw_error("xorec_init() must be called before calling xorec_encode()");
    if (xorec_check_args(block_size, data_blks_per_chunk, parity_blks_per_chunk) != XorecResult::Success) return XorecResult::InvalidCounts;
    if (block_size % sizeof(CUDA_ATOMIC_XOR_T) != 0) return XorecResult::InvalidSize;

    cudaMemset(parity_buffer, 0, block_size * parity_blks_per_chunk * num_chunks);
    xorec_encode_kernel_full_message<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(data_buffer, parity_buffer, num_chunks, block_size, data_blks_per_chunk, parity_blks_per_chunk);
  
    return XorecResult::Success;
  }

  XorecResult xorec_gpu_encode_full_message_cpu_parity(
    const uint8_t *XOREC_RESTRICT data_buffer,
    uint8_t *XOREC_RESTRICT parity_buffer,
    uint8_t *XOREC_RESTRICT gpu_parity_buffer,
    size_t num_chunks,
    size_t block_size,
    size_t data_blks_per_chunk,
    size_t parity_blks_per_chunk
  ) {
    if (!XOREC_GPU_INIT_CALLED) throw_error("xorec_init() must be called before calling xorec_encode()");
    if (xorec_check_args(block_size, data_blks_per_chunk, parity_blks_per_chunk) != XorecResult::Success) return XorecResult::InvalidCounts;
    if (block_size % sizeof(CUDA_ATOMIC_XOR_T) != 0) return XorecResult::InvalidSize;

    cudaMemset(gpu_parity_buffer, 0, block_size * parity_blks_per_chunk * num_chunks);
    xorec_encode_kernel_full_message<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(data_buffer, gpu_parity_buffer, num_chunks, block_size, data_blks_per_chunk, parity_blks_per_chunk);
    cudaMemcpyAsync(parity_buffer, gpu_parity_buffer, block_size * parity_blks_per_chunk * num_chunks, cudaMemcpyDeviceToHost);
    return XorecResult::Success;
  }
  
  __global__ void xorec_encode_kernel_full_message(
    const uint8_t * XOREC_RESTRICT data_buffer,
    uint8_t * XOREC_RESTRICT parity_buffer,
    size_t num_chunks,
    size_t block_size,
    size_t data_blks_per_chunk,
    size_t parity_blks_per_chunk
  ) {
    unsigned num_threads = blockDim.x * gridDim.x;
    unsigned glbl_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  
    unsigned tot_elems = num_chunks * data_blks_per_chunk * block_size / sizeof(CUDA_ATOMIC_XOR_T);
    unsigned chunk_elems = data_blks_per_chunk * block_size / sizeof(CUDA_ATOMIC_XOR_T);
    unsigned block_elems = block_size / sizeof(CUDA_ATOMIC_XOR_T);

  
    for (unsigned i = glbl_thread_idx; i < tot_elems; i += num_threads) {
      unsigned chunk_idx = i / chunk_elems;
      unsigned block_idx = (i % chunk_elems) / block_elems;
      unsigned parity_idx = block_idx % parity_blks_per_chunk;
  
      const CUDA_ATOMIC_XOR_T * XOREC_RESTRICT data_block_64 = reinterpret_cast<const CUDA_ATOMIC_XOR_T*>(
        data_buffer
        + chunk_idx * data_blks_per_chunk * block_size
        + block_idx * block_size
      );
      CUDA_ATOMIC_XOR_T * XOREC_RESTRICT parity_block_64 = reinterpret_cast<CUDA_ATOMIC_XOR_T*>(
        parity_buffer
        + chunk_idx * parity_blks_per_chunk * block_size
        + parity_idx * block_size
      );
  
      atomicXor(&parity_block_64[(i % chunk_elems) % block_elems], data_block_64[(i % chunk_elems) % block_elems]);
    } 
}