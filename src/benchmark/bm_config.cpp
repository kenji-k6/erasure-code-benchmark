#include "bm_config.hpp"
#define KiB *1024
# define MiB *1024*1024

const std::vector<size_t> VAR_DATA_SIZES = {
  128 KiB, 256 KiB, 512 KiB, 1 MiB, 2 MiB
};

const std::vector<ECTuple> VAR_EC_PARAMS = {
  { 4+32, 32 },     { 8+32, 32 },     { 16+32, 32 },
  { 8+64, 64 },     { 16+64, 64 },    { 32+64, 64 },
  { 16+128, 128 },  { 32+128, 128 },  { 64+128, 128 }
};

const std::vector<size_t> VAR_NUM_LOST_BLOCKS = {
  0, 1, 2, 4, 8, 16, 32, 64
};

const std::vector<size_t> VAR_NUM_GPU_BLOCKS = {
  8, 16, 32, 64, 128, 256
};

const std::vector<size_t> VAR_NUM_THREADS_PER_BLOCK = {
  32, 64, 128, 256, 512
};