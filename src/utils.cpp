#include "utils.h"

/*
 * PCGRandom number generator, used to generate random data for benchmarking
*/

PCGRandom::PCGRandom(uint64_t seed, uint64_t seq) {
  state = 0;
  inc = (seq << 1) | 1; // Ensure inc is odd
  next();
  state += seed;
  next();
}

uint32_t PCGRandom::next() {
  uint64_t prev_state = state;
  state = (prev_state * 6364136223846793005ull) + inc;
  uint32_t xorshifted = static_cast<uint32_t>(((prev_state >> 18u) ^ prev_state) >> 27u);
  uint32_t rotated = static_cast<uint32_t>(prev_state >> 59u);
  return (xorshifted >> rotated) | (xorshifted << ((-rotated) & 31));
}


void set_block_check_values(
  uint8_t* block_ptr,
  size_t block_size,
  size_t block_index
) {
  PCGRandom rng(RNG_SEED+block_index, 1);
  for (unsigned i = 0; i < block_size; i++) {
    block_ptr[i] = rng.next() % 256;
  }
}


bool check_block_for_corruption(
  uint8_t* block_ptr,
  size_t block_size,
  size_t block_index
) {
  PCGRandom rng(RNG_SEED+block_index, 1);
  for (unsigned i = 0; i < block_size; i++) {
    if (block_ptr[i] != (rng.next() % 256)) {
      return true;
    }
  }
  return false;
}
