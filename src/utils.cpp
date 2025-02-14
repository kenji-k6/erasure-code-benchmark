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



static UTIL_FORCE_INLINE void* simd_safe_allocate(size_t size) {
  uint8_t* data = (uint8_t *) calloc(1, ALIGNMENT_BYTES + size);

  if (!data) return nullptr;
  
  unsigned offset = (unsigned)((uintptr_t)data % ALIGNMENT_BYTES);
  data += ALIGNMENT_BYTES - offset;
  data[-1] = (uint8_t)offset;
  return (void*)data;
}

static UTIL_FORCE_INLINE void simd_safe_free(void* ptr) {
  if (!ptr) return;
  uint8_t* data = (uint8_t*)ptr;
  unsigned offset = data[-1];
  if (offset >= ALIGNMENT_BYTES) exit(1); // should never happen
  data -= ALIGNMENT_BYTES - offset;
  free(data);
}
