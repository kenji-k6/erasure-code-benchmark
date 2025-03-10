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


LeopardBenchmark::LeopardBenchmark(const BenchmarkConfig& config) noexcept : ECBenchmark(config) {
  // Initialize Leopard
  if (leo_init()) throw_error("Leopard: Initialization failed.");

  encode_work_count_ = leo_encode_work_count(num_original_blocks_, num_recovery_blocks_);
  decode_work_count_ = leo_decode_work_count(num_original_blocks_, num_recovery_blocks_);

  if (encode_work_count_ == 0 || decode_work_count_ == 0) throw_error("Leopard: Invalid work count(s).");

  // Allocate buffers
  original_buffer_ = std::make_unique<uint8_t[]>(block_size_ * num_original_blocks_);
  encode_buffer_ = std::make_unique<uint8_t[]>(block_size_ * encode_work_count_);
  decode_buffer_ = std::make_unique<uint8_t[]>(block_size_ * decode_work_count_);

  if (!original_buffer_ || !encode_buffer_ || !decode_buffer_) throw_error("Leopard: Failed to allocate buffer(s).");

  // Populate vectors with pointers to the data blocks
  original_ptrs_.resize(num_original_blocks_);
  encode_work_ptrs_.resize(encode_work_count_);
  decode_work_ptrs_.resize(decode_work_count_);

  for (unsigned i = 0; i < num_original_blocks_; ++i) original_ptrs_[i] = &original_buffer_[i * block_size_];
  for (unsigned i = 0; i < encode_work_count_; ++i) encode_work_ptrs_[i] = &encode_buffer_[i * block_size_];
  for (unsigned i = 0; i < decode_work_count_; ++i) decode_work_ptrs_[i] = &decode_buffer_[i * block_size_];

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < num_original_blocks_; ++i) {
    int write_res = write_validation_pattern(i, original_ptrs_[i], block_size_);
    if (write_res) throw_error("Leopard: Failed to write random checking packet.");
  }
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
  for (auto idx : lost_block_idxs_) {
    if (idx < num_original_blocks_) {
      memset(original_ptrs_[idx], 0, block_size_);
      original_ptrs_[idx] = nullptr;
    } else {
      memset(encode_work_ptrs_[idx - num_original_blocks_], 0, block_size_);
      encode_work_ptrs_[idx- num_original_blocks_] = nullptr;
    }
  }
}


bool LeopardBenchmark::check_for_corruption() const noexcept {
  for (unsigned i = 0; i < num_original_blocks_; i++) {
    if (!original_ptrs_[i]) { // lost block
      if (!validate_block(decode_work_ptrs_[i], block_size_)) return false;
    } else { // block is intact
      if (!validate_block(original_ptrs_[i], block_size_)) return false;
    }
  }
  return true;
}
