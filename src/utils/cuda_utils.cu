#include "cuda_utils.cuh"
#include "utils.hpp"

/**
 * @attention COMMENTED IMPLEMENTATION IS INCORRECT
 */

__global__ void touch_memory_kernel(const uint8_t* buffer, size_t size) { 
  // uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  // uint32_t num_threads = gridDim.x * blockDim.x;
  // uint32_t bytes_per_thread = (size + num_threads - 1) / num_threads;

  // for (uint32_t i = idx*bytes_per_thread; i < (idx+1)*bytes_per_thread; i++) {
  //   if (i < size) {
  //     [[maybe_unused]] volatile uint8_t val = buffer[i]; // Use volatile to prevent compiler optimizations
  //   }
  // }
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    [[maybe_unused]] volatile uint8_t val = buffer[idx]; // Use volatile to prevent compiler optimizations
  }
}

__host__ void touch_memory(const uint8_t* buffer, size_t size) {
  // uint32_t device_id = 0;
  // cudaError_t err = cudaSetDevice(device_id);
  // if (err != cudaSuccess) throw_error("Failed to set device");
  
  // cudaDeviceProp device_prop;
  // cudaGetDeviceProperties(&device_prop, device_id);

  // uint32_t max_threads_per_block = device_prop.maxThreadsPerBlock;
  // // Launch the kernel
  // touch_memory_kernel<<<1, max_threads_per_block>>>(buffer, size);

  // err = cudaDeviceSynchronize();
  // if (err != cudaSuccess) throw_error("CUDA error in touch_memory: " + std::string(cudaGetErrorString(err)));

  int threads_per_block = 256;
  int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

  // Launch the kernel
  touch_memory_kernel<<<blocks_per_grid, threads_per_block>>>(buffer, size);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA error in touch_memory: " + std::string(cudaGetErrorString(err)));
  }
}