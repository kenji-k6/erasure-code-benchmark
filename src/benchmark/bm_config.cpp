#include "bm_config.hpp"
#define KiB *1024

const std::vector<size_t> VAR_BLOCK_SIZES = {
  4 KiB, 8 KiB, 16 KiB, 32 KiB, 64 KiB, 128 KiB, 256 KiB
};

const std::vector<ECTuple> VAR_EC_PARAMS = {
  { 1+16, 16 },   { 2+16, 16},      { 4+16, 16 },     { 8+16, 16 },
  { 2+32, 32 },   { 4+32, 32 },     { 8+32, 32 },     { 16+32, 32 },
  { 4+64, 64 },   { 8+64, 64 },     { 16+64, 64 },    { 32+64, 64 },
  { 8+128, 128 }, { 16+128, 128 },  { 32+128, 128 },  { 64+128, 128 }
};

const std::vector<size_t> VAR_NUM_LOST_BLOCKS = {
  0, 1, 2, 4, 8, 16, 32, 64
};

const std::vector<size_t> VAR_NUM_GPU_BLOCKS = {
  8, 16, 32, 64, 128, 256, 512, 1024
};

const std::vector<size_t> VAR_NUM_THREADS_PER_BLOCK = {
  32, 64, 128, 256, 512, 1024
};