#include "bm_config.hpp"

const std::vector<size_t> VAR_BLOCK_SIZES = { 65536 };
const std::vector<FECTuple> VAR_FEC_PARAMS = { {32, 8} };
const std::vector<size_t> VAR_NUM_CPU_THREADS = { 1, 2, 4, 8, 16, 32, 48, 60 };