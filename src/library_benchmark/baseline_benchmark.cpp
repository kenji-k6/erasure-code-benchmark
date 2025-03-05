#include "baseline_benchmark.h"
#include "xorbaseline.h"
#include "utils.h"
#include <cstring>


BaselineBenchmark::BaselineBenchmark(const BenchmarkConfig& config) noexcept : ECBenchmark(config) {}

int BaselineBenchmark::setup() noexcept {
  // Allocate buffers with proper alignment for SIMD
  data_buffer_ = static_cast<uint8_t*>(aligned_alloc(ALIGNMENT_BYTES, block_size_ * num_original_blocks_));
  parity_buffer_ = static_cast<uint8_t*>(aligned_alloc(ALIGNMENT_BYTES, block_size_ * num_recovery_blocks_));

  if (!data_buffer_ || !parity_buffer_) {
    std::cerr << "Baseline: Failed to allocate buffer(s).\n";
    teardown();
    return -1;
  }
  
  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < num_original_blocks_; ++i) {
    int write_res = write_validation_pattern(i, data_buffer_ + i * block_size_, block_size_);
    if (write_res) {
      std::cerr << "Baseline: Failed to write random checking packet.\n";
      teardown();
      return -1;
    }
  }
  
  return 0;
}

void BaselineBenchmark::teardown() noexcept {
  if (data_buffer_) free(data_buffer_);
  if (parity_buffer_) free(parity_buffer_);
}

int BaselineBenchmark::encode() noexcept {
  xor_encode(data_buffer_, parity_buffer_, block_size_, num_original_blocks_, num_recovery_blocks_);
  return 0;
}

int BaselineBenchmark::decode() noexcept {
  xor_decode(data_buffer_, parity_buffer_, block_size_, num_original_blocks_, num_recovery_blocks_, block_bitmap_);
  return 0;
}

void BaselineBenchmark::simulate_data_loss() noexcept {
  uint32_t loss_idx = 0;
  for (unsigned i = 0; i < num_original_blocks_ + num_recovery_blocks_; ++i) {
    if (loss_idx < num_lost_blocks_ && lost_block_idxs_[loss_idx] == i) {
      if (i < num_original_blocks_) {
        memset(data_buffer_ + i * block_size_, 0, block_size_);
      } else {
        memset(parity_buffer_ + (i - num_original_blocks_) * block_size_, 0, block_size_);
      }

      ++loss_idx;
      continue;
    }
    block_bitmap_.set(i);
  }
}

bool BaselineBenchmark::check_for_corruption() const noexcept {
  for (unsigned i = 0; i < num_original_blocks_; ++i) {
    if (!validate_block(data_buffer_ + i * block_size_, block_size_)) return false;
  }
  return true;
}