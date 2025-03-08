#include "cuda_utils.cuh"
#include <stdexcept>


__global__ void touch_memory_kernel(const uint8_t* buffer, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    [[maybe_unused]] volatile uint8_t val = buffer[idx]; // Use volatile to prevent compiler optimizations
  }
}

void touch_memory(const uint8_t* buffer, size_t size) {
  int threads_per_block = 256;
  int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

  // Launch the kernel
  touch_memory_kernel<<<blocks_per_grid, threads_per_block>>>(buffer, size);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA error in touch_memory: " + std::string(cudaGetErrorString(err)));
  }
}