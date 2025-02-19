#include "cm256_benchmark.h"

#include "utils.h"
#include <iostream>


/**
 * @file cm256_benchmark.cpp
 * @brief Benchmark implementation for the CM256 ECC Library
 * 
 * Documentation can be found in cm256_benchmark.h and abstract_benchmark.h
 */


int CM256Benchmark::setup() noexcept {
  // Store variables used in performance-critical areas locally
  num_original_blocks_ = benchmark_config.computed.num_original_blocks;
  num_recovery_blocks_ = benchmark_config.computed.num_recovery_blocks;
  block_size_ = benchmark_config.block_size;

  // Initialize CM256
  if (cm256_init()) {
    std::cerr << "CM256: Initialization failed.\n";
    return -1;
  }

  // Initialize CM256 parameters
  params_.BlockBytes = block_size_;
  params_.OriginalCount = num_original_blocks_;
  params_.RecoveryCount = num_recovery_blocks_;

  // Allocate  buffers with proper alignment for SIMD
  original_buffer_ = static_cast<uint8_t*>(aligned_alloc(ALIGNMENT_BYTES, block_size_ * num_original_blocks_));
  decode_buffer_ = static_cast<uint8_t*>(aligned_alloc(ALIGNMENT_BYTES, block_size_ * num_recovery_blocks_));

  // Resize block vector
  blocks_.resize(ECCLimits::CM256_MAX_TOT_BLOCKS);

  if (!original_buffer_ || !decode_buffer_) {
    std::cerr << "CM256: Failed to allocate buffer(s).\n";
    teardown();
    return -1;
  }

  // Initialize block array content
  for (unsigned i = 0; i < num_original_blocks_; i++) {
    blocks_[i].Block = original_buffer_ + i * block_size_;
    blocks_[i].Index = cm256_get_original_block_index(params_, i);
  }

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < num_original_blocks_; i++) {
    int write_res = write_validation_pattern(i, static_cast<uint8_t*>(blocks_[i].Block), block_size_);
    if (write_res) {
      std::cerr << "CM256: Failed to write random checking packet.\n";
      teardown();
      return -1;
    }
  }
  
  return 0;
}


void CM256Benchmark::teardown() noexcept {
  if (original_buffer_) free(original_buffer_);
  if (decode_buffer_) free(decode_buffer_);
}


int CM256Benchmark::encode() noexcept {
  return cm256_encode(params_, blocks_.data(), static_cast<void*>(decode_buffer_));
}


int CM256Benchmark::decode() noexcept {
  return cm256_decode(params_, blocks_.data());
}


void CM256Benchmark::simulate_data_loss() noexcept {
  for (unsigned i = 0; i < benchmark_config.num_lost_blocks; i++) {
    uint32_t idx = lost_block_idxs[i];
    if (idx < num_original_blocks_) { // dropped block is original block
      idx = cm256_get_original_block_index(params_, idx);
      memset(original_buffer_ + idx * block_size_, 0, block_size_);
      blocks_[idx].Block = decode_buffer_ + idx * block_size_;
      blocks_[idx].Index = cm256_get_recovery_block_index(params_, idx);
    } else { // dropped block is recovery block
      uint32_t orig_idx = idx - num_original_blocks_;
      memset(decode_buffer_ + orig_idx * block_size_, 0, block_size_);
    }
  }
}


bool CM256Benchmark::check_for_corruption() const noexcept {
  for (unsigned i = 0; i < num_original_blocks_; i++) {
    if (!validate_block(static_cast<uint8_t*>(blocks_[i].Block), block_size_)) return false;
  }
  return true;
}


void CM256Benchmark::flush_cache() noexcept {
  // TODO: Implement cache flushing
}