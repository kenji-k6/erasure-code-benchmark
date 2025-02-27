#include "baseline_benchmark.h"
#include "utils.h"
#include <cstring>


BaselineBenchmark::BaselineBenchmark(const BenchmarkConfig& config) noexcept : ECCBenchmark(config) {}

int BaselineBenchmark::setup() noexcept {
  // Allocate buffers with proper alignment for SIMD
  original_buffer_ = static_cast<uint8_t*>(aligned_alloc(ALIGNMENT_BYTES, block_size_ * num_original_blocks_));
  parity_block_ = static_cast<uint8_t*>(aligned_alloc(ALIGNMENT_BYTES, block_size_));

  if (!original_buffer_ || !parity_block_) {
    std::cerr << "Baseline: Failed to allocate buffer(s).\n";
    teardown();
    return -1;
  }
  
  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < num_original_blocks_; ++i) {
    int write_res = write_validation_pattern(i, original_buffer_ + i * block_size_, block_size_);
    if (write_res) {
      std::cerr << "Baseline: Failed to write random checking packet.\n";
      teardown();
      return -1;
    }
  }
  
  return 0;
}

void BaselineBenchmark::teardown() noexcept {
  if (original_buffer_) free(original_buffer_);
  if (parity_block_) free(parity_block_);
}

int BaselineBenchmark::encode() noexcept {
  /// @attention For this to be efficient, the block size has to be a multple of 8 byte
  // Zero out parity block
  memset(parity_block_, 0, block_size_);

  // XOR all data blocks to get parity block
  for (unsigned i = 0; i < num_original_blocks_; ++i) {
    for (unsigned j = 0; j < block_size_; j += 8) {
      *((uint64_t*)(parity_block_ + j)) ^= *((uint64_t*)(original_buffer_ + i * block_size_ + j));
    }
  }
  return 0;
}

int BaselineBenchmark::decode() noexcept {
  /// @attention For this to be efficient, the block size has to be a multple of 8 byte
  if (num_lost_blocks_ != 1) return 0;

  // XOR the parity block with all received blocks to recover the lost block
  for (unsigned i = 0; i < num_original_blocks_; ++i) {
    for (unsigned j = 0; j < block_size_; j += 8) {
      *((uint64_t*)(parity_block_ + j)) ^= *((uint64_t*)(original_buffer_ + i * block_size_ + j));
    }
  }

  memcpy(original_buffer_ + lost_block_idxs_[0] * block_size_, parity_block_, block_size_);
  return 0;
}

void BaselineBenchmark::simulate_data_loss() noexcept {
  if (num_lost_blocks_ == 0) return;
  uint32_t idx = lost_block_idxs_[0];
  memset(original_buffer_ + idx * block_size_, 0, block_size_);
}

bool BaselineBenchmark::check_for_corruption() const noexcept {
  for (unsigned i = 0; i < num_original_blocks_; ++i) {
    if (!validate_block(original_buffer_ + i * block_size_, block_size_)) return false;
  }
  return true;
}