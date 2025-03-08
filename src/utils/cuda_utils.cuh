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
void touch_memory(const uint8_t* buffer, size_t size);

#endif // CUDA_UTILS_CUH