#include "isal_benchmark.h"

int ISALBenchmark::setup() {

  // Allocate buffers
  original_data_ = (uint8_t*) simd_safe_allocate(kConfig.data_size);
  if (!original_data_) {
    std::cerr << "ISAL: Failed to allocate original data buffer.\n";
    return -1;
  }

  recovery_data_ = (uint8_t*) simd_safe_allocate(kConfig.computed.recovery_blocks * kConfig.data_size);
  if (!recovery_data_) {
    teardown();
    std::cerr << "ISAL: Failed to allocate recovery data buffer.\n";
    return -1;
  }
  
  // Allocate pointers
  original_data_ptrs_ = new uint8_t*[kConfig.computed.original_blocks];
  recovery_data_ptrs_ = new uint8_t*[kConfig.computed.recovery_blocks];


  for (unsigned i = 0; i < kConfig.computed.original_blocks; i++) {
    original_data_ptrs_[i] = (uint8_t *) (original_data_ + (i * kConfig.block_size));
  }

  for (unsigned i = 0; i < kConfig.computed.recovery_blocks; i++) {
    recovery_data_ptrs_[i] = (uint8_t *) (recovery_data_ + (i * kConfig.block_size));
  }

  // Initialize original data to 1s, recovery data to 0s
  memset(original_data_, 0xFF, kConfig.data_size);
  memset(recovery_data_, 0, kConfig.computed.recovery_blocks * kConfig.block_size);

}