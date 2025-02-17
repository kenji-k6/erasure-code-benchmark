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
  original_buffer_ = (uint8_t*) simd_safe_allocate(kConfig.data_size);
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

  // Initialze original data to 1s, recovery data to 0s
  memset(original_buffer_, 0xFF, kConfig.data_size);
  memset(recovery_buffer_, 0, kConfig.block_size * kConfig.computed.recovery_blocks);

  // Initialize blocks
  for (unsigned i = 0; i < kConfig.computed.original_blocks; i++) {
    blocks_[i].Block = original_buffer_ + (i * kConfig.block_size);
    blocks_[i].Index = cm256_get_original_block_index(params_, i);
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
  return false;
}



void CM256Benchmark::simulate_data_loss() {
  // TODO: Implement data loss simulation
}