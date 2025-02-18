#include "cm256_benchmark.h"

int CM256Benchmark::setup() noexcept {
  // Initialize cm256
  if (cm256_init()) {
    std::cerr << "CM256: Initialization failed.\n";
    return -1;
  }

  // Initialize cm256 parameter struct
  params_.BlockBytes = benchmark_config.block_size;
  params_.OriginalCount = benchmark_config.computed.num_original_blocks;
  params_.RecoveryCount = benchmark_config.computed.num_recovery_blocks;


  // Allocate buffers
  original_buffer_ = (uint8_t*) aligned_alloc(ALIGNMENT_BYTES, benchmark_config.block_size * benchmark_config.computed.num_original_blocks);

  if (!original_buffer_) {
    teardown(); 
    std::cerr << "CM256: Failed to allocate original buffer.\n";
    return -1;
  }

  recovery_buffer_ = (uint8_t*) aligned_alloc(ALIGNMENT_BYTES, benchmark_config.block_size * benchmark_config.computed.num_recovery_blocks);
  if (!recovery_buffer_) {
    teardown();
    std::cerr << "CM256: Failed to allocate recovery buffer.\n";
    return -1;
  }

  // Initialize blocks
  for (unsigned i = 0; i < benchmark_config.computed.num_original_blocks; i++) {
    blocks_[i].Block = original_buffer_ + (i * benchmark_config.block_size);
    blocks_[i].Index = cm256_get_original_block_index(params_, i);
  }

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < benchmark_config.computed.num_original_blocks; i++) {
    int write_res = write_validation_pattern(
      i,
      (uint8_t *) blocks_[i].Block,
      benchmark_config.block_size
    );

    if (write_res) {
      teardown();
      std::cerr << "CM256: Failed to write random checking packet.\n";
      return -1;
    }
  }

  return 0;
}



void CM256Benchmark::teardown() noexcept {
  if (original_buffer_) free(original_buffer_);
  if (recovery_buffer_) free(recovery_buffer_);
}



int CM256Benchmark::encode() noexcept {
  // Encode the data
  return cm256_encode(params_, blocks_, (void*) recovery_buffer_);
}



int CM256Benchmark::decode() noexcept {
  // Decode the data
  return cm256_decode(params_, blocks_);
}



void CM256Benchmark::flush_cache() noexcept {
  // TODO: Implement cache flushing
}



bool CM256Benchmark::check_for_corruption() const noexcept {
  for (int i = 0; i < benchmark_config.computed.num_original_blocks; i++) {
    if (!validate_block((uint8_t *)blocks_[i].Block, benchmark_config.block_size)) return false;
  }
  return true;
}



void CM256Benchmark::simulate_data_loss() noexcept {
  
  for (unsigned i = 0; i < benchmark_config.num_lost_blocks; i++) {
    uint32_t idx = lost_block_idxs[i];
    // if idx >= benchmark_config.computed.num_original_blocks, then it is a recovery block
    // else it's a data block

    if (idx < benchmark_config.computed.num_original_blocks) {
      // Zero out memory
      memset(original_buffer_ + (idx * benchmark_config.block_size), 0, benchmark_config.block_size);
      blocks_[idx].Block = recovery_buffer_ + (benchmark_config.block_size * idx);
      blocks_[idx].Index = cm256_get_recovery_block_index(params_, idx);
      


    } else {
      uint32_t orig_idx = idx-benchmark_config.computed.num_original_blocks;
      memset(recovery_buffer_ + (orig_idx * benchmark_config.block_size), 0, benchmark_config.block_size);
    }
  }
}