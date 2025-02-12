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
  uint64_t prevState = state;
  state = (prevState * 6364136223846793005ull) + inc;
  uint32_t xorshifted = static_cast<uint32_t>(((prevState >> 18u) ^ prevState) >> 27u);
  uint32_t rotated = static_cast<uint32_t>(prevState >> 59u);
  return (xorshifted >> rotated) | (xorshifted << ((-rotated) & 31));
}