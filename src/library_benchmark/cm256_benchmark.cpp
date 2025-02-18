#include "cm256_benchmark.h"

int CM256Benchmark::setup() {
  // Initialize cm256
  if (cm256_init()) {
    std::cerr << "CM256: Initialization failed.\n";
    return -1;
  }

  // Initialize cm256 parameter struct
  params_.BlockBytes = kConfig.block_size;
  params_.OriginalCount = kConfig.computed.original_blocks;
  params_.RecoveryCount = kConfig.computed.recovery_blocks;


  // Allocate buffers
  original_buffer_ = (uint8_t*) simd_safe_allocate(kConfig.block_size * kConfig.computed.original_blocks);

  if (!original_buffer_) {
    teardown(); 
    std::cerr << "CM256: Failed to allocate original buffer.\n";
    return -1;
  }

  recovery_buffer_ = (uint8_t*) simd_safe_allocate(kConfig.block_size * kConfig.computed.recovery_blocks);
  if (!recovery_buffer_) {
    teardown();
    std::cerr << "CM256: Failed to allocate recovery buffer.\n";
    return -1;
  }

  // Initialize blocks
  for (unsigned i = 0; i < kConfig.computed.original_blocks; i++) {
    blocks_[i].Block = original_buffer_ + (i * kConfig.block_size);
    blocks_[i].Index = cm256_get_original_block_index(params_, i);
  }

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < kConfig.computed.original_blocks; i++) {
    int write_res = write_random_checking_packet(
      i,
      (uint8_t *) blocks_[i].Block,
      kConfig.block_size
    );

    if (write_res) {
      teardown();
      std::cerr << "CM256: Failed to write random checking packet.\n";
      return -1;
    }
  }

  return 0;
}



void CM256Benchmark::teardown() {
  if (original_buffer_) simd_safe_free(original_buffer_);
  if (recovery_buffer_) simd_safe_free(recovery_buffer_);
}



int CM256Benchmark::encode() {
  // Encode the data
  return cm256_encode(params_, blocks_, (void*) recovery_buffer_);
}



int CM256Benchmark::decode() {
  // Decode the data
  return cm256_decode(params_, blocks_);
}



void CM256Benchmark::flush_cache() {
  // TODO: Implement cache flushing
}



bool CM256Benchmark::check_for_corruption() {
  for (int i = 0; i < kConfig.computed.original_blocks; i++) {
    if (!check_packet((uint8_t *)blocks_[i].Block, kConfig.block_size)) return false;
  }
  return true;
}



void CM256Benchmark::simulate_data_loss() {
  
  for (unsigned i = 0; i < kConfig.num_lost_blocks; i++) {
    uint32_t idx = kLost_block_idxs[i];
    // if idx >= kConfig.computed.original_blocks, then it is a recovery block
    // else it's a data block

    if (idx < kConfig.computed.original_blocks) {
      // Zero out memory
      memset(original_buffer_ + (idx * kConfig.block_size), 0, kConfig.block_size);
      blocks_[idx].Block = recovery_buffer_ + (kConfig.block_size * idx);
      blocks_[idx].Index = cm256_get_recovery_block_index(params_, idx);
      


    } else {
      uint32_t orig_idx = idx-kConfig.computed.original_blocks;
      memset(recovery_buffer_ + (orig_idx * kConfig.block_size), 0, kConfig.block_size);
    }
  }
}