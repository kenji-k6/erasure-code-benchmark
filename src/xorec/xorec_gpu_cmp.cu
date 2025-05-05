#include "xorec_gpu_cmp.cuh"
#include "utils.hpp"
#include <iostream>

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
  size_t num_chunks,
  size_t block_size,
  size_t chunk_data_blocks,
  size_t chunk_parity_blocks,
  size_t num_gpu_blocks,
  size_t threads_per_block
) {
  if (!XOREC_GPU_INIT_CALLED) throw_error("xorec_gpu_init() must be called before calling xorec_encode()");
  XorecResult err = xorec_check_args(data_buf, parity_buf, block_size, chunk_data_blocks, chunk_parity_blocks);
  if (err != XorecResult::Success) return err;
  if (block_size % sizeof(CUDA_ATOMIC_XOR_T) != 0) return XorecResult::InvalidSize;

  cudaMemsetAsync(parity_buf, 0, num_chunks * chunk_parity_blocks * block_size);
  xorec_gpu_xor_kernel<<<num_gpu_blocks, threads_per_block>>>(
    data_buf,
    parity_buf,
    num_chunks,
    block_size,
    chunk_data_blocks,
    chunk_parity_blocks
  );

  return XorecResult::Success;
}



XorecResult xorec_gpu_decode(
  uint8_t* XOREC_RESTRICT data_buf,
  uint8_t* XOREC_RESTRICT parity_buf,
  size_t num_chunks,
  size_t block_size,
  size_t chunk_data_blocks,
  size_t chunk_parity_blocks,
  const uint8_t* XOREC_RESTRICT block_bitmap,
  size_t num_gpu_blocks,
  size_t threads_per_block
) {
  if (!XOREC_GPU_INIT_CALLED) throw_error("xorec_gpu_init() must be called before calling xorec_encode()");

  XorecResult err = xorec_check_args(data_buf, parity_buf, block_size, chunk_data_blocks, chunk_parity_blocks);
  if (err != XorecResult::Success) return err;
  if (block_size % sizeof(CUDA_ATOMIC_XOR_T) != 0) return XorecResult::InvalidSize;
  
  bool recover_required = false;
  for (unsigned c = 0; c < num_chunks; ++c) {
    auto chunk_bitmap = block_bitmap + c * (chunk_data_blocks + chunk_parity_blocks);
    auto chunk_data_buf = data_buf + c * chunk_data_blocks * block_size;
    
    if (require_recovery(chunk_data_blocks, chunk_bitmap)) recover_required = true;
    if (!is_recoverable(chunk_data_blocks, chunk_parity_blocks, chunk_bitmap)) return XorecResult::DecodeFailure;
    // Zero out lost blocks
    for (unsigned i = 0; i < chunk_data_blocks; ++i) {
      if (!chunk_bitmap[i]) cudaMemsetAsync(chunk_data_buf + i * block_size, 0, block_size);
    }
  }

  if (!recover_required) return XorecResult::Success;

  xorec_gpu_xor_kernel<<<num_gpu_blocks, threads_per_block>>>(
    data_buf,
    parity_buf,
    num_chunks,
    block_size,
    chunk_data_blocks,
    chunk_parity_blocks
  );

  // copy recovered blocks back to data_buf
  for (unsigned c = 0; c < num_chunks; ++c) {
    auto chunk_bitmap = block_bitmap + c * (chunk_data_blocks + chunk_parity_blocks);
    auto chunk_data_buf = data_buf + c * chunk_data_blocks * block_size;
    auto chunk_parity_buf = parity_buf + c * chunk_parity_blocks * block_size;

    for (unsigned i = 0; i < chunk_data_blocks; ++i) {
      if (!chunk_bitmap[i]) cudaMemcpyAsync(chunk_data_buf + i * block_size, chunk_parity_buf + (i % chunk_parity_blocks) * block_size, block_size, cudaMemcpyDeviceToDevice);
    }
  }

  return XorecResult::Success;
}



__global__ void xorec_gpu_xor_kernel(
  const uint8_t* XOREC_RESTRICT data_buf,
  uint8_t* XOREC_RESTRICT parity_buf,
  size_t num_chunks,
  size_t block_size,
  size_t chunk_data_blocks,
  size_t chunk_parity_blocks
) {
  unsigned num_threads = blockDim.x * gridDim.x;
  unsigned glbl_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned tot_elems = num_chunks * chunk_data_blocks * block_size / sizeof(CUDA_ATOMIC_XOR_T);
  unsigned chunk_elems = chunk_data_blocks * block_size / sizeof(CUDA_ATOMIC_XOR_T);
  unsigned block_elems = block_size / sizeof(CUDA_ATOMIC_XOR_T);

  for (unsigned i = glbl_thread_idx; i < tot_elems; i += num_threads) {
    unsigned chunk_idx = i / chunk_elems;
    unsigned block_idx = (i % chunk_elems) / block_elems;
    unsigned parity_idx = block_idx % chunk_parity_blocks;

    const CUDA_ATOMIC_XOR_T * XOREC_RESTRICT data_block = reinterpret_cast<const CUDA_ATOMIC_XOR_T*>(
      data_buf + (chunk_idx * chunk_data_blocks * block_size) + (block_idx * block_size)
    );
    CUDA_ATOMIC_XOR_T * XOREC_RESTRICT parity_block = reinterpret_cast<CUDA_ATOMIC_XOR_T*>(
      parity_buf + (chunk_idx * chunk_parity_blocks * block_size) + (parity_idx * block_size)
    );

    atomicXor(&parity_block[(i%chunk_elems)%block_elems], data_block[(i%chunk_elems)%block_elems]);
  }
}


