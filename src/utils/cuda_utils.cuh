#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <cstdint>

/**
 * @brief CUDA kernel to "touch" the memory buffer.acoshf32x
 * 
 * @param buffer Pointer to the buffer to be touched
 * @param size Size of the buffer in bytes
 */
__global__ void touch_memory_kernel(const uint8_t* buffer, size_t size);


/**
 * @brief Helper function to launch the `touch_memory_kernel`
 * 
 * This function is called from CPU code to ensure the buffer is "cold" in CPU memory.
 * 
 * @param buffer Pointer to the buffer to be touched.
 * @param size Size of the buffer in bytes.
 */
__host__ void touch_memory(const uint8_t* buffer, size_t size);


/**
 * @brief Helper function to allocated aligned, unified memory
 * @attention to free you have to use the corresponding aligned free (see below)
 */

__host__ cudaError_t aligned_cudaMallocManaged(void** devPtr, size_t size, unsigned int flags = cudaMemAttachGlobal, size_t alignment);

__host__ cudaError_t aligned_cudaFree(void* devPtr);
#endif // CUDA_UTILS_CUH