/**
 * @file utils.cpp
 * 
 * @attention Documentation can be found in utils.h
 */

 
#include "utils.hpp"

#include <iostream>
#include <numeric>
#include <chrono>



// PCGRandom class implementation
PCGRandom::PCGRandom(uint64_t seed, uint64_t seq)
  : state(0),
    inc((seq << 1) | 1) // Ensure inc is odd 
{
  next();         // Generate initial state
  state += seed;  // Apply seed
  next();         // Generate another random value after seeding
}

uint32_t PCGRandom::next() {
  uint64_t prev_state = state;
  state = (prev_state * 6364136223846793005ULL) + inc; // Linear congruential formula
  uint32_t xorshifted = static_cast<uint32_t>(((prev_state >> 18U) ^ prev_state) >> 27U);
  uint32_t rotated = static_cast<uint32_t>(prev_state >> 59U);
  return (xorshifted >> rotated) | (xorshifted << ((-rotated) & 31));
}

// Utility function: Write validation pattern for data corrution detection
int write_validation_pattern(uint8_t* block_ptr, size_t bytes) {
  #if ENABLE_VALIDATION
  if (bytes < 2) return -1; // Invalid block size

  const auto now = std::chrono::system_clock::now(); // used as seed for random number generator
  const uint64_t time_seed = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  PCGRandom rng(RANDOM_SEED+time_seed, 1);

  if (bytes < 16) { // For small blocks, CRC is not viable
    uint8_t val = static_cast<uint8_t>(rng.next());
    std::fill(block_ptr, block_ptr + bytes, val);
  } else {
    uint32_t crc = static_cast<uint32_t>(bytes);
    // Store the block size for later verification
    *reinterpret_cast<uint32_t*>(block_ptr + 4) = bytes;

    for (unsigned i = 8; i < bytes; ++i) {
      uint8_t val = static_cast<uint8_t>(rng.next());
      block_ptr[i] = val;
      crc = (crc << 3) | (crc >> (32 - 3)); // Spread entropy
      crc += val;
    }

    // Store the computed CRC at the beginning of the block
    *reinterpret_cast<uint32_t*>(block_ptr) = crc;
  }
  #else
  // Simulate a write, so that compiler doesn't optimize it away
  PCGRandom rng(RANDOM_SEED, 1);
  uint8_t val = static_cast<uint8_t>(rng.next());
  std::fill(block_ptr, block_ptr + bytes, val);
  asm volatile("" : "+m"(*block_ptr) : : "memory");
  #endif
  return 0;
}

// Utility function: Validate block content integrity
bool validate_block(const uint8_t* block_ptr, size_t bytes) {
  #if ENABLE_VALIDATION
  if (bytes < 2) return false; // Invalid block size

  if (bytes < 16) { // No CRC, check for uniform data
    uint8_t val = block_ptr[0];
    return std::all_of(block_ptr + 1, block_ptr + bytes, [val](uint8_t b) { return b == val; });
  }else {
    const uint32_t read_bytes = *reinterpret_cast<const uint32_t*>(block_ptr + 4);
    if (read_bytes != bytes) return false;

    uint32_t crc = static_cast<uint32_t>(bytes);

    // Recompute CRC
    for (unsigned i = 8; i < bytes; ++i) {
      const uint8_t val = block_ptr[i];
      crc = (crc << 3) | (crc >> (32 - 3));
      crc += val;
    }
    const uint32_t block_crc = *reinterpret_cast<const uint32_t*>(block_ptr); // the actual stored CRC
    return block_crc == crc;
  }
  #else
  return true; // Validation disabled
  #endif
}

// Utility function: Select lost blocks, ensuring the set is recoverable for Xorec
void select_lost_blocks(size_t num_data_blocks, size_t num_parity_blocks, size_t num_lost_blocks, uint8_t* block_bitmap) {
  if (num_lost_blocks == 0) return;
  if (num_lost_blocks > num_parity_blocks) {
    std::cerr << "select_lost_blocks: Number of lost blocks must be less than or equal to the number of recovery blocks\n";
    std::exit(EXIT_SUCCESS);
  }

  const size_t tot_blocks = num_data_blocks + num_parity_blocks;
  const auto now = std::chrono::system_clock::now(); // used as seed for random number generator
  const uint64_t time_seed = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  PCGRandom rng(RANDOM_SEED+time_seed, 1);


  // Vector of indices that can be removed, while maintaing recoverability
  // of the lost blocks
  std::vector<uint32_t> valid_idxs(tot_blocks);
  std::iota(valid_idxs.begin(), valid_idxs.end(), 0);

  for (uint32_t i = 0; i < num_lost_blocks; ++i) {
    const size_t idx = rng.next() % valid_idxs.size();
    const uint32_t lost_idx = valid_idxs[idx];
    block_bitmap[lost_idx] = 0; // Mark the block as lost

    const uint32_t parity_class = lost_idx % num_parity_blocks; // Compute Xorec parity class
    // Remove all blocks in the same parity class from valid indices
    std::erase_if(valid_idxs, [parity_class, num_parity_blocks](uint32_t idx) { return (idx % num_parity_blocks) == parity_class; });
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