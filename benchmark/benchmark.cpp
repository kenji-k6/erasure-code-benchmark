#include "benchmark.h"



// Leopard Benchmark implementation
class LeopardBenchmark : public ECCBenchmark {
public:
  int setup(const BenchmarkConfig& config) override { 
    config_ = config;

    // Assert that the block size is a multiple of 64 bytes
    if (config_.computed.actual_block_size % LEOPARD_BLOCK_SIZE_ALIGNMENT != 0) {
      std::cerr << "Leopard: Block size must be a multiple of " << LEOPARD_BLOCK_SIZE_ALIGNMENT << " bytes.\n";
      return -1;
    }

    // Assert that the number of blocks is within the valid range
    if (config_.computed.original_blocks < LEOPARD_MIN_BLOCKS || config_.computed.original_blocks > LEOPARD_MAX_BLOCKS) {
      std::cerr << "Leopard: Original blocks must be between " << LEOPARD_MIN_BLOCKS << " and " << LEOPARD_MAX_BLOCKS << ".\n";
      return -1;
    }

    // Initialize Leopard
    if (leo_init()) {
      std::cerr << "Leopard: Initialization failed.\n";
      return -1;
    }

    // Compute encode work count
    encode_work_count_ = leo_encode_work_count(config_.computed.original_blocks, config_.computed.recovery_blocks);
    if (encode_work_count_ == 0) {
      std::cerr << "Leopard: Invalid encode work count.\n";
      return -1;
    }

    // Compute decode work count
    decode_work_count_ = leo_decode_work_count(config_.computed.original_blocks, config_.computed.recovery_blocks);
    if (decode_work_count_ == 0) {
      std::cerr << "Leopard: Invalid decode work count.\n";
      return -1;
    }

    // Allocate memory for buffers
    original_ptrs_.resize(config_.computed.original_blocks);
    recovery_ptrs_.resize(config_.computed.recovery_blocks);
    work_ptrs_.resize(encode_work_count_);

    // Allocate memory for original data
    const size_t block_size = config_.computed.actual_block_size;
    for (size_t i = 0; i < config_.computed.original_blocks; i++) {
      original_ptrs_[i] = malloc(config_.computed.actual_block_size);
      if (!original_ptrs_[i]) {
        std::cerr << "Leopard: Failed to allocate memory for original data block " << i << ".\n";
        teardown();
        return -1; 
      }

      memset(original_ptrs_[i], 1, block_size); // Initialize to 1
    }

    // Allocate memory for recovery data
    for (size_t i = 0; i < config_.computed.recovery_blocks; i++) {
      recovery_ptrs_[i] = malloc(config_.computed.actual_block_size);
      if (!recovery_ptrs_[i]) {
        std::cerr << "Leopard: Failed to allocate memory for recovery data block " << i << ".\n";
        teardown();
        return -1; 
      }

      memset(recovery_ptrs_[i], 0, block_size); // Initialize to 0
    }

    // Allocate memory for work data
    for (size_t i = 0; i < encode_work_count_; i++) {
      work_ptrs_[i] = malloc(config_.computed.actual_block_size);
      if (!work_ptrs_[i]) {
        std::cerr << "Leopard: Failed to allocate memory for work data block " << i << ".\n";
        teardown();
        return -1; 
      }

      memset(work_ptrs_[i], 0, block_size); // Initialize to 0
    }
    return 0;
  }

private:
  unsigned encode_work_count_ = 0;
  unsigned decode_work_count_ = 0;
  std::vector<void*> original_ptrs_;
  std::vector<void*> recovery_ptrs_;
  std::vector<void*> work_ptrs_;
  long long encode_time_us_ = 0;
  long long decode_time_us_ = 0;
  size_t memory_usage_ = 0;
  double throughput_mbps_ = 0;
  BenchmarkConfig config_;
}; // class LeopardBenchmark