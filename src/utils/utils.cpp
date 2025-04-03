/**
 * @file utils.cpp
 * @brief Implements utility functions for benchmarking, random number generation, and data corruption detection.
 * 
 * Documentation can be found in utils.h
 */

 
#include "utils.hpp"

#include <iostream>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cuda_runtime.h>

// PCGRandom class implementation
PCGRandom::PCGRandom(uint64_t seed, uint64_t seq) {
  state = 0;
  inc = (seq << 1) | 1; // Ensure increment is odd
  next();               // Generate initial state
  state += seed;        // Apply seed
  next();               // Generate another random value after seeding
}


uint32_t PCGRandom::next() {
  uint64_t prev_state = state;
  state = (prev_state * 6364136223846793005ULL) + inc; // Linear congruential formula
  uint32_t xorshifted = static_cast<uint32_t>(((prev_state >> 18U) ^ prev_state) >> 27U);
  uint32_t rotated = static_cast<uint32_t>(prev_state >> 59U);
  return (xorshifted >> rotated) | (xorshifted << ((-rotated) & 31));
}


#ifdef VALIDATION
int write_validation_pattern(uint32_t block_idx, uint8_t* block_ptr, size_t bytes) {
  if (bytes < 2) throw_error("write_validation_pattern: num_bytes must be at least 2");
  
  PCGRandom rng(block_idx, 1);

  if (bytes < 16) { // CRC check not viable
    uint8_t val = static_cast<uint8_t>(rng.next());
    std::fill(block_ptr, block_ptr + bytes, val);
  } else {
    // Apply CRC to the block if it is large enough
    uint32_t crc = bytes;
    *reinterpret_cast<uint32_t*>(block_ptr + 4) = bytes;

    for (unsigned i = 8; i < bytes; i++) {
      uint8_t val = static_cast<uint8_t>(rng.next());
      block_ptr[i] = val;
      crc = (crc << 3) | (crc >> (32 - 3)); // Spread entropy
      crc += val;
    }

    *reinterpret_cast<uint32_t*>(block_ptr) = crc;
  } 
  return 0;
}
#else
// Utility functions
int write_validation_pattern(uint32_t block_idx, uint8_t* block_ptr, size_t bytes) {
  // Original Implementation isn't needed, because we aren't decoding/losing data
  // The code below ensures that the compiler doesn't optimize away memeory accesses
  
  PCGRandom rng(block_idx, 1);
  uint8_t val = static_cast<uint8_t>(rng.next());
  std::fill(block_ptr, block_ptr + bytes, val);
  asm volatile ("" : : : "memory");
  return 0;
}

#endif

bool validate_block(const uint8_t* block_ptr, size_t bytes) {
  if (bytes < 2) return false; // Invalid block size

  if (bytes < 16) { // No CRC, check for uniform data
    uint8_t val = block_ptr[0];
    return std::all_of(block_ptr + 1, block_ptr + bytes, [val](uint8_t b) { return b == val; });
  }else {
    uint32_t crc = bytes;
    uint32_t read_bytes = *reinterpret_cast<const uint32_t*>(block_ptr + 4);
    if (read_bytes != bytes) return false;

    // Recompute CRC
    for (unsigned i = 8; i < bytes; i++) {
      uint8_t val = block_ptr[i];
      crc = (crc << 3) | (crc >> (32 - 3));
      crc += val;
    }

    uint32_t block_crc = *reinterpret_cast<const uint32_t*>(block_ptr); // the actual stored CRC
    return block_crc == crc;
  }
}

void select_lost_block_idxs(size_t num_data_blocks, size_t num_parity_blocks, size_t num_lost_blocks, uint8_t* block_bitmap) {
  if (num_lost_blocks > num_parity_blocks) {
    std::cerr << "select_lost_block_idxs: Number of lost blocks must be less than or equal to the number of recovery blocks\n";
    exit(0);
  }

  size_t tot_blocks = num_data_blocks + num_parity_blocks;

  auto now = std::chrono::system_clock::now(); // used as seed for random number generator
  uint64_t time_seed = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

  PCGRandom rng(RANDOM_SEED+time_seed, 1);
  // Vector of valid indices to remove
  std::vector<uint32_t> valid_idxs(tot_blocks);
  std::iota(valid_idxs.begin(), valid_idxs.end(), 0);

  for (uint32_t i = 0; i < num_lost_blocks; i++) {
    uint32_t lost_idx = valid_idxs[rng.next()%valid_idxs.size()];
    block_bitmap[lost_idx] = 0; // Mark the block as lost
    uint32_t recovery_set = lost_idx % num_parity_blocks;
    
    // update valid indices
    for (auto it = valid_idxs.begin(); it != valid_idxs.end();) {
      it = (*it % num_parity_blocks == recovery_set) ? valid_idxs.erase(it) : it+1;
    }
  }
}


[[noreturn]] void throw_error(const std::string& message) {
  throw std::runtime_error(message);
}

std::string to_lower(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(),
    [](unsigned char c){ return std::tolower(c); });
  return str;
}