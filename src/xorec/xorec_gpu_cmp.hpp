#ifndef XOREC_GPU_CMP_HPP
#define XOREC_GPU_CMP_HPP

extern uint32_t DEVICE_ID;
extern uint32_t MAX_THREADS_PER_BLOCK;
extern uint32_t MAX_THREADS_PER_MULTIPROCESSOR;
extern uint32_t MAX_BLOCKS_PER_MULTIPROCESSOR;
extern uint32_t WARP_SIZE;
extern bool XOREC_GPU_INIT_CALLED;

void xorec_gpu_init();

#endif // XOREC_GPU_CMP_HPP