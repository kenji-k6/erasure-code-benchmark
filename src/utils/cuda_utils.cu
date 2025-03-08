#include "cuda_utils.cuh"
#include <stdexcept>
#include <cassert>


__global__ void touch_memory_kernel(const uint8_t* buffer, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    [[maybe_unused]] volatile uint8_t val = buffer[idx]; // Use volatile to prevent compiler optimizations
  }
}

__host__ void touch_memory(const uint8_t* buffer, size_t size) {
  int threads_per_block = 256;
  int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

  // Launch the kernel
  touch_memory_kernel<<<blocks_per_grid, threads_per_block>>>(buffer, size);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA error in touch_memory: " + std::string(cudaGetErrorString(err)));
  }
}



__host__ cudaError_t aligned_cudaMallocManaged(void** devPtr, size_t size, size_t alignment, unsigned int flags) {
  assert((alignment & (alignment - 1)) == 0); // Ensure alignment is a power of 2
  size_t extra = alignment - 1 + sizeof(void*);
  void* unaligned_ptr = nullptr;
  
  cudaError_t err = cudaMallocManaged(&unaligned_ptr, size + extra, flags);

  if (err != cudaSuccess) return err;

  void* aligned_ptr = reinterpret_cast<void*>((reinterpret_cast<size_t>(unaligned_ptr) + extra) & ~(alignment - 1));

  // Store the original pointer before the aligned pointer, so we can free it later
  reinterpret_cast<void**>(aligned_ptr)[-1] = unaligned_ptr;

  *devPtr = aligned_ptr;
  return cudaSuccess;
}


__host__ cudaError_t aligned_cudaFree(void* devPtr) {
  if (!devPtr) return cudaSuccess;
  void *unaligned_ptr = reinterpret_cast<void**>(devPtr)[-1];
  return cudaFree(unaligned_ptr);
}