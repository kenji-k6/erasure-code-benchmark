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




static int write_random_checking_packet(
  size_t block_idx,
  uint8_t* block_ptr,
  uint32_t num_bytes 
) {
  PCGRandom rng(block_idx, 1);

  if (num_bytes < 16) { // not enough room for crc check
    if (num_bytes < 2) {
      std::cerr << "write_random_checking_packet: num_bytes must be at least 2\n";
      return -1;
    }
    // Fill all bytes with same random number
    block_ptr[0] = (uint8_t) rng.next();

    for (unsigned i = 1; i < num_bytes; i++) {
      block_ptr[i] = (uint8_t) rng.next();
    }
  } else {
    // if we have >= 16 bytes we use a cyclic redundancy check
    uint32_t crc = num_bytes;
    *(uint32_t*)(block_ptr+4) = num_bytes;

    for (unsigned i = 8; i < num_bytes; i++) {
      uint8_t val = (uint8_t) rng.next();
      block_ptr[i] = val;
      crc = (crc << 3) | (crc >> (32 - 3)); // shifting to spread entropy
      crc += val;
    }

    *(uint32_t *) block_ptr = crc;
  }
  return 0;
}


static bool check_packet(
  uint8_t* block_ptr,
  uint32_t num_bytes
) {
  if (num_bytes < 16) { // We didn't compute a crc
    if (num_bytes < 2) return false;
    uint8_t val = block_ptr[0];
    for (unsigned i = 1; i < num_bytes; i++) {
      if (block_ptr[i] != val) {
        return false;
      }
    }
  } else {
    uint32_t crc = num_bytes;
    uint32_t read_bytes = *(uint32_t *)(block_ptr+4);

    if (read_bytes != num_bytes) {
      return false;
    }

    // Recompute CRC
    for (unsigned i = 8; i < num_bytes; i++) {
      uint8_t val = block_ptr[i];
      crc = (crc << 3) | (crc >> (32 - 3));
      crc += val;
    }

    uint32_t block_crc = *(uint32_t *) block_ptr; // the actual stored crc

    // check if CRC's match
    if (block_crc != crc) {
      return false;
    }
  }

  return true;
}
