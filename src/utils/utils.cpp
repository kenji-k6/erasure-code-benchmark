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
  // if (bytes < 2) throw_error("write_validation_pattern: num_bytes must be at least 2");
  
  PCGRandom rng(block_idx, 1);

  // if (bytes < 16) { // CRC check not viable
  //   uint8_t val = static_cast<uint8_t>(rng.next());
  //   std::fill(block_ptr, block_ptr + bytes, val);
  // } else {
  //   // Apply CRC to the block if it is large enough
  //   uint32_t crc = bytes;
  //   *reinterpret_cast<uint32_t*>(block_ptr + 4) = bytes;

  //   for (unsigned i = 8; i < bytes; i++) {
  //     uint8_t val = static_cast<uint8_t>(rng.next());
  //     block_ptr[i] = val;
  //     crc = (crc << 3) | (crc >> (32 - 3)); // Spread entropy
  //     crc += val;
  //   }

  //   *reinterpret_cast<uint32_t*>(block_ptr) = crc;
  // } 


  // Original Implementation isn't needed, because we aren't decoding/losing data
  // The code below ensures that the compiler doesn't optimize away memeory accesses
  uint8_t val = static_cast<uint8_t>(rng.next());
  std::fill(block_ptr, block_ptr + bytes, val);
  asm volatile ("" : : : "memory");
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


[[noreturn]] void throw_error(const std::string& message) {
  throw std::runtime_error(message);
}

std::string to_lower(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(),
    [](unsigned char c){ return std::tolower(c); });
  return str;
}