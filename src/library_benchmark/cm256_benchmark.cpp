#include "cm256_benchmark.h"

int CM256Benchmark::setup(const BenchmarkConfig& config) {
  config_ = config;

  if (config_.block_size < CM256_MIN_BLOCKS || config.block_size > CM256_MAX_BLOCKS) {
    std::cerr << "CM256: Block size must be between " << CM256_MIN_BLOCKS << " and " << CM256_MAX_BLOCKS << " (is " << config_.block_size << ").\n";
    return -1;
  }

  if (config_.computed.recovery_blocks > CM256_MAX_BLOCKS-config_.computed.original_blocks) {
    std::cerr << "CM256: Recovery blocks must be between 0 and " << CM256_MAX_BLOCKS-config_.computed.original_blocks << " (is " << config_.computed.recovery_blocks << ").\n";
    return -1;
  }

  if (cm256_init()) {
    std::cerr << "CM256: Initialization failed.\n";
    return -1;
  }

  // Initialize cm256 parameter struct
  params_.BlockBytes = config_.block_size;
  params_.OriginalCount = config_.computed.original_blocks;
  params_.RecoveryCount = config_.computed.recovery_blocks;


  // Initialize original data
  original_data_ = (uint8_t*) malloc(config_.data_size);
  if (!original_data_) {
    std::cerr << "CM256: Failed to allocate memory for original data.\n";
    teardown();
    return -1;
  }

  // Initialize to 1
  memset(original_data_, 1, config_.data_size);

  // Initialize original block pointers
  for (int i = 0; i < config_.computed.original_blocks; i++) {
    blocks_[i].Block = original_data_ + (i * config_.block_size);
    blocks_[i].Index = cm256_get_original_block_index(params_, i);
  }

  // Allocate memory recovery data
  recovery_data_ = (uint8_t*) malloc(config_.block_size * config_.computed.recovery_blocks);
  if (!recovery_data_) {
    std::cerr << "CM256: Failed to allocate memory for recovery data.\n";
    teardown();
    return -1;
  }
  
  return 0;
}


int CM256Benchmark::encode() {
  // TODO: Implement

  return -1;
}


int CM256Benchmark::decode(double loss_rate) {
  // TODO: Implement
  return -1;
}


void CM256Benchmark::teardown() {
  // TODO: Implement
  return;
}


ECCBenchmark::Metrics CM256Benchmark::get_metrics() const {
  // TODO: Implement
  return {0,0,0,0,0,0};
}