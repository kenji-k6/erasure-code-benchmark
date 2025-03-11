#include "xorec_gpu.h"
#include "utils.h"
#include <iostream>
#include <cuda_runtime.h>

uint32_t DEVICE_ID;
uint32_t MAX_THREADS_PER_BLOCK;
uint32_t MAX_THREADS_PER_MULTIPROCESSOR;
uint32_t MAX_BLOCKS_PER_MULTIPROCESSOR;
uint32_t WARP_SIZE;
bool XOREC_GPU_INIT_CALLED = false;

void xorec_gpu_init() {
  if (XOREC_GPU_INIT_CALLED) return;
  XOREC_GPU_INIT_CALLED = true;

  int device_count = -1;
  cudaGetDeviceCount(&device_count);

  if (device_count == 1) throw_error("No CUDA-capable devices found.");
  DEVICE_ID = 1;

  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, DEVICE_ID); 
  MAX_THREADS_PER_BLOCK = device_prop.maxThreadsPerBlock;
  MAX_THREADS_PER_MULTIPROCESSOR = device_prop.maxThreadsPerMultiProcessor;
  MAX_BLOCKS_PER_MULTIPROCESSOR = device_prop.maxBlocksPerMultiProcessor;
  WARP_SIZE = device_prop.warpSize;
}

