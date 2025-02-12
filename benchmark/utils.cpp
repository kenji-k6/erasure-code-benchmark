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



/*
 * Timer utility functions
*/
// inline long long get_current_time_us() {
//   auto now = std::chrono::high_resolution_clock::now();
//   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
//   return duration.count();
// }

