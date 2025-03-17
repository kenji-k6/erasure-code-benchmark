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


// Utility functions
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


void select_lost_block_idxs(size_t num_recovery_blocks, size_t num_lost_blocks, uint32_t max_idx, std::vector<uint32_t>& lost_block_idxs) {
  if (num_lost_blocks > num_recovery_blocks) {
    std::cerr << "select_lost_block_idxs: Number of lost blocks must be less than or equal to the number of recovery blocks\n";
    exit(0);
  }

  auto now = std::chrono::system_clock::now(); // used as seed for random number generator
  uint64_t time_seed = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

  PCGRandom rng(RANDOM_SEED+time_seed, 1);
  // Vector of valid indices to remove
  std::vector<uint32_t> valid_idxs(max_idx);
  std::iota(valid_idxs.begin(), valid_idxs.end(), 0);

  for (uint32_t i = 0; i < num_lost_blocks; i++) {
    lost_block_idxs.push_back(valid_idxs[rng.next()%valid_idxs.size()]);
    uint32_t recovery_set = lost_block_idxs[i] % num_recovery_blocks;
    
    // update valid indices
    for (auto it = valid_idxs.begin(); it != valid_idxs.end();) {
      if (*it % num_recovery_blocks == recovery_set) {
        it = valid_idxs.erase(it);
      } else {
        ++it;
      }
    }
  }

  // Sort the indices (needed for Wirehair and ISA-L)
  std::sort(lost_block_idxs.begin(), lost_block_idxs.end());
}


[[noreturn]] void throw_error(const std::string& message) {
  throw std::runtime_error(message);
}

std::string to_lower(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(),
    [](unsigned char c){ return std::tolower(c); });
  return str;
}

void touch_memory(uint8_t* buffer, size_t bytes) {
  int device_id = 0;
  cudaError_t err = cudaMemPrefetchAsync(buffer, bytes, device_id);
  if (err != cudaSuccess) throw_error("touch_memory: cudaMemPrefetchAsync failed");
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) throw_error("touch_memory: cudaDeviceSynchronize failed");
}