#ifndef XOREC_GPU_H
#define XOREC_GPU_H
#include <cstdint>

extern uint32_t DEVICE_ID;
extern uint32_t MAX_THREADS_PER_BLOCK;
extern uint32_t MAX_THREADS_PER_MULTIPROCESSOR;
extern uint32_t MAX_BLOCKS_PER_MULTIPROCESSOR;
extern uint32_t WARP_SIZE;

void xorec_gpu_init();
#endif // XOREC_GPU_H