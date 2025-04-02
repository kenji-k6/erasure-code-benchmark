#include "bm_config.hpp"

const std::vector<size_t> VAR_BLOCK_SIZES = { 4096/*, 8192, 16384, 32768, 65536, 131072, 262144 */};
const std::vector<FECTuple> VAR_FEC_PARAMS = {/* {2,1}, {4,2}, {8,4}, {16,4}, {16,8}, {32,4}, */{32, 8} };
const std::vector<size_t> VAR_NUM_CPU_THREADS = { 1, 2, 4, 8, 16, 32 };
const std::vector<size_t> VAR_NUM_GPU_BLOCKS = { 2048, 1024, 512, 256, 128, 64, 32, 16, 8 };
const std::vector<size_t> VAR_THREADS_PER_GPU_BLOCK = { 32, 64, 128, 256, 512, 1024 };