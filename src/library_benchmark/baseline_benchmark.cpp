#include "baseline_benchmark.h"
#include "utils.h"

int BaselineBenchmark::setup() noexcept {
  auto num_original_blocks_ = benchmark_config.computed.num_original_blocks;
  auto num_recovery_blocks_ = benchmark_config.computed.num_recovery_blocks;
  auto block_size_ = benchmark_config.block_size;

  baseline_init();
  
  uint8_t *original_buffer_ = static_cast<uint8_t*>(aligned_alloc(ALIGNMENT_BYTES, block_size_ * num_original_blocks_));
  uint8_t *recovery_buffer_ = static_cast<uint8_t*>(aligned_alloc(ALIGNMENT_BYTES, block_size_ * num_recovery_blocks_));

  if (!original_buffer_ || !recovery_buffer_) {
    std::cerr << "Leopard: Failed to allocate buffer(s).\n";
    teardown();
    return -1;
  }

  params_ = baseline_get_params(num_original_blocks_, num_recovery_blocks_, block_size_, original_buffer_, recovery_buffer_);

  InvMatPtr_ = static_cast<uint8_t*>(aligned_alloc(ALIGNMENT_BYTES, sizeof(uint32_t)*256*256));

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < num_original_blocks_; i++) {
    int write_res = write_validation_pattern(i, original_buffer_ + block_size_*i, block_size_);
    if (write_res) {
      std::cerr << "Leopard: Failed to write random checking packet.\n";
      teardown();
      return -1;
    }
  }
  
  return 0;
}

  



int BaselineBenchmark::encode() noexcept {
  baseline_encode(params_);
  return 0;
}



int BaselineBenchmark::decode() noexcept {
  baseline_decode(params_, reinterpret_cast<uint32_t*>(InvMatPtr_), benchmark_config.num_lost_blocks, lost_block_idxs.data());
  return 0;
}




bool BaselineBenchmark::check_for_corruption() const noexcept {
  for (unsigned i = 0; i < benchmark_config.computed.num_original_blocks; i++) {
    if (!validate_block(static_cast<uint8_t*>(params_.orig_data) + i * benchmark_config.block_size, benchmark_config.block_size)) return false; 
  }

  return true;
}



void BaselineBenchmark::simulate_data_loss() noexcept {
  for (unsigned i = 0; i < benchmark_config.num_lost_blocks; i++) {
    uint32_t idx = lost_block_idxs[i];

    if (idx < benchmark_config.computed.num_original_blocks) {
      // Zero out the block in the original data array, set the corresponding block pointer to nullptr
      memset(static_cast<uint8_t*>(params_.orig_data) + idx * benchmark_config.block_size, 0, benchmark_config.block_size);
    } else {
      idx -= 127;
      // Zero out the block in the encoded data array, set the corresponding block pointer to nullptr
      memset(static_cast<uint8_t*>(params_.redundant_data) + idx * benchmark_config.block_size, 0, benchmark_config.block_size);
    }
  }
}



void BaselineBenchmark::teardown() noexcept {

}

void BaselineBenchmark::flush_cache() noexcept {
  
}