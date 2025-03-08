/**
 * @file leopard_benchmark.cpp
 * @brief Benchmark implementation for the Leopard EC library
 * 
 * Documentation can be found in leopard_benchmark.h and abstract_benchmark.h
 */


#include "leopard_benchmark.h"

#include "leopard.h"
#include "utils.h"
#include <cstring>
#include <iostream>


LeopardBenchmark::LeopardBenchmark(const BenchmarkConfig& config) noexcept : ECBenchmark(config) {}

int LeopardBenchmark::setup() noexcept {
  
  // Initialize Leopard
  if (leo_init()) {
    std::cerr << "Leopard: Initialization failed.\n";
    return -1;
  }

  // Compute encode/decode work counts
  encode_work_count_ = leo_encode_work_count(num_original_blocks_, num_recovery_blocks_);
  decode_work_count_ = leo_decode_work_count(num_original_blocks_, num_recovery_blocks_);

  if (encode_work_count_ == 0 || decode_work_count_ == 0) {
    std::cerr << "Leopard: Invalid work count(s): encode=" << encode_work_count_ << ", decode=" << decode_work_count_ << "\n";
    return -1;
  }

  // Allocate buffer with proper alignment for SIMD
  original_buffer_ = static_cast<uint8_t*>(aligned_alloc(ALIGNMENT_BYTES, block_size_ * num_original_blocks_));
  encode_buffer_ = static_cast<uint8_t*>(aligned_alloc(ALIGNMENT_BYTES, block_size_ * encode_work_count_));
  decode_buffer_ = static_cast<uint8_t*>(aligned_alloc(ALIGNMENT_BYTES, block_size_ * decode_work_count_));

  if (!original_buffer_ || !encode_buffer_ || !decode_buffer_) {
    std::cerr << "Leopard: Failed to allocate buffer(s).\n";
    teardown();
    return -1;
  }

  // Allocate pointers to the data blocks
  original_ptrs_.resize(num_original_blocks_);
  encode_work_ptrs_.resize(encode_work_count_);
  decode_work_ptrs_.resize(decode_work_count_);

  // Initialize pointers to appropriate memory locations in the buffer
  for (unsigned i = 0; i < num_original_blocks_; i++) {
    original_ptrs_[i] = original_buffer_ + i * block_size_;
  }
  for (unsigned i = 0; i < encode_work_count_; i++) {
    encode_work_ptrs_[i] = encode_buffer_ + i * block_size_;
  }
  for (unsigned i = 0; i < decode_work_count_; i++) {
    decode_work_ptrs_[i] = decode_buffer_ + i * block_size_;
  }

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < num_original_blocks_; i++) {
    int write_res = write_validation_pattern(i, original_ptrs_[i], block_size_);
    if (write_res) {
      std::cerr << "Leopard: Failed to write random checking packet.\n";
      teardown();
      return -1;
    }
  }

  return 0;
}


void LeopardBenchmark::teardown() noexcept {
  // Free manually allocated memory
  if (original_buffer_) free(original_buffer_);
  if (encode_buffer_) free(encode_buffer_);
  if (decode_buffer_) free(decode_buffer_);
}


int LeopardBenchmark::encode() noexcept {
  return leo_encode(block_size_, num_original_blocks_,
                    num_recovery_blocks_, encode_work_count_,
                    reinterpret_cast<void**>(original_ptrs_.data()),
                    reinterpret_cast<void**>(encode_work_ptrs_.data()));
}


int LeopardBenchmark::decode() noexcept {
  return leo_decode(block_size_, num_original_blocks_,
                    num_recovery_blocks_, decode_work_count_,
                    reinterpret_cast<void**>(original_ptrs_.data()),
                    reinterpret_cast<void**>(encode_work_ptrs_.data()),
                    reinterpret_cast<void**>(decode_work_ptrs_.data()));
}


void LeopardBenchmark::simulate_data_loss() noexcept {
  for (unsigned i = 0; i < num_lost_blocks_; i++) {
    uint32_t idx = lost_block_idxs_[i];
    if (idx < num_original_blocks_) {
      // Zero out the block in the original data array, set the corresponding block pointer to nullptr
      memset(original_ptrs_[idx], 0, block_size_);
      original_ptrs_[idx] = nullptr;
    } else {
      idx -= num_original_blocks_;
      // Zero out the block in the encoded data array, set the corresponding block pointer to nullptr
      memset(encode_work_ptrs_[idx], 0, block_size_);
      encode_work_ptrs_[idx] = nullptr;
    }
  }
}


bool LeopardBenchmark::check_for_corruption() const noexcept {
  for (unsigned i = 0; i < num_original_blocks_; i++) {
    bool res = false;
    if (!original_ptrs_[i]) { // lost block
      res = validate_block(decode_work_ptrs_[i], block_size_);
    } else { // block is intact
      res = validate_block(original_ptrs_[i], block_size_);
    }

    if (!res) return false;
  }

  return true;
}
