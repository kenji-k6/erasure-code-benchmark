#include "bm_config.hpp"

const std::vector<size_t> VAR_BLOCK_SIZES = {
  1 KiB, 2 KiB, 4 KiB, 8 KiB
};

const std::vector<ECTuple> VAR_EC_PARAMS = {
  { 4+8,  8 },
  { 4+16, 16 }, { 8+16, 16 },
  { 4+32, 32 }, { 8+32, 32 }
};

const std::vector<size_t> VAR_NUM_CPU_THREADS = {
  1, 2, 4, 8, 16, 32
};

const std::vector<size_t> VAR_NUM_LOST_BLOCKS = {
  0, 1, 2, 4, 8
};

const std::vector<size_t> VAR_NUM_GPU_BLOCKS = { 256 };

const std::vector<size_t> VAR_NUM_THREADS_PER_BLOCK = { 512 };