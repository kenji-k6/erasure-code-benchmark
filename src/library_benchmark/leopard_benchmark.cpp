#include "leopard_benchmark.h"

int LeopardBenchmark::setup(const BenchmarkConfig& config) {
  config_ = config;

  // Assert that the block size is a multiple of 64 bytes
  if (config_.block_size % LEOPARD_BLOCK_SIZE_ALIGNMENT != 0) {
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
  encode_work_ptrs_.resize(encode_work_count_);
  decode_work_ptrs_.resize(decode_work_count_);

  // Allocate memory for original data
  for (size_t i = 0; i < config_.computed.original_blocks; i++) {
    original_ptrs_[i] = malloc(config_.block_size);
    if (!original_ptrs_[i]) {
      std::cerr << "Leopard: Failed to allocate memory for original data block " << i << ".\n";
      teardown();
      return -1; 
    }

    memset(original_ptrs_[i], 1, config_.block_size); // Initialize to 1
  }

  // Allocate memory for encode work data
  for (size_t i = 0; i < encode_work_count_; i++) {
    encode_work_ptrs_[i] = malloc(config_.block_size);
    if (!encode_work_ptrs_[i]) {
      std::cerr << "Leopard: Failed to allocate memory for work data block " << i << ".\n";
      teardown();
      return -1; 
    }

    memset(encode_work_ptrs_[i], 0, config_.block_size); // Initialize to 0
  }

  // Allocate memory for decode work data
  for (size_t i = 0; i < decode_work_count_; i++) {
    decode_work_ptrs_[i] = malloc(config_.block_size);
    if (!decode_work_ptrs_[i]) {
      std::cerr << "Leopard: Failed to allocate memory for recovery data block " << i << ".\n";
      teardown();
      return -1; 
    }

    memset(decode_work_ptrs_[i], 0, config_.block_size); // Initialize to 0
  }

  // Calculate memory usage
  memory_used_ = (config_.computed.original_blocks
                  + encode_work_count_
                  + decode_work_count_
                  ) * config_.block_size;
  
  // Calculate total data bytes
  total_data_bytes_ = config_.computed.original_blocks * config_.block_size;                

  return 0;
}


int LeopardBenchmark::encode() {
  // Start the timer
  auto start_time = std::chrono::high_resolution_clock::now();

  // Encode the data
  LeopardResult encode_result = leo_encode(
    config_.block_size,
    config_.computed.original_blocks,
    config_.computed.recovery_blocks,
    encode_work_count_,
    (void**) &original_ptrs_[0],
    (void**) &encode_work_ptrs_[0]
  );

  // Stop the timer
  auto end_time = std::chrono::high_resolution_clock::now();

  // Check for errors
  if (encode_result != Leopard_Success) {
    std::cerr << "Leopard: Encode failed with error " << encode_result << ".\n";
    return -1;
  }

  // Calculate the time taken to encode
  encode_time_us_ = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

  // Calculate the throughput(s)
  encode_input_throughput_mbps_ = ((double) (total_data_bytes_ * 8)) / encode_time_us_; // throughput of (original) input data
  encode_output_throughput_mbps_ = ((double) (config_.computed.recovery_blocks * config_.block_size * 8)) / encode_time_us_; // throughput of output data (recovery blocks)

  return 0;
}


int LeopardBenchmark::decode(double loss_rate) {
  // Start the timer
  auto start_time = std::chrono::high_resolution_clock::now();
  // Simulate data loss
  // TODO: Implement data loss simulation

  // Decode the data
  LeopardResult decode_result = leo_decode(
    config_.block_size,
    config_.computed.original_blocks,
    config_.computed.recovery_blocks,
    decode_work_count_,
    (void**) &original_ptrs_[0],
    (void**) &encode_work_ptrs_[0],
    (void**) &decode_work_ptrs_[0]
  );

  // Stop the timer
  auto end_time = std::chrono::high_resolution_clock::now();

  // Check for errors in decoding
  if (decode_result != Leopard_Success) {
    std::cerr << "Leopard: Decode failed with error " << decode_result << ".\n";
    return -1;
  }

  // Check for corruption
  // TODO: Implement data corruption check

  // Calculate the time taken to decode
  decode_time_us_ = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

  // Calculate the throughput(s)
  decode_input_throughput_mbps_ = ((double) (total_data_bytes_ * 8)) / decode_time_us_; // throughput of (original) input data
  decode_output_throughput_mbps_ = ((double) (total_data_bytes_ * config_.loss_rate * 8)) / decode_time_us_; // throughput of lost bit recovery

  return 0;
}


void LeopardBenchmark::teardown() {
  // Free allocated memory
  for (auto ptr : original_ptrs_) {
    if (ptr) free(ptr);
  }

  for (auto ptr : encode_work_ptrs_) {
    if (ptr) free(ptr);
  }

  for (auto ptr : decode_work_ptrs_) {
    if (ptr) free(ptr);
  }

  // TODO: Check if these clears are needed
  original_ptrs_.clear();
  encode_work_ptrs_.clear();
  decode_work_ptrs_.clear();

  // Reset the benchmark state
  encode_time_us_ = 0;
  decode_time_us_ = 0;
  memory_used_ = 0;
  encode_input_throughput_mbps_ = 0.0;
  encode_output_throughput_mbps_ = 0.0;
  decode_input_throughput_mbps_ = 0.0;
  decode_output_throughput_mbps_ = 0.0;
}


ECCBenchmark::Metrics LeopardBenchmark::get_metrics() const {
  return {
    encode_time_us_,
    decode_time_us_,
    memory_used_,
    encode_input_throughput_mbps_,
    encode_output_throughput_mbps_,
    decode_input_throughput_mbps_,
    decode_output_throughput_mbps_
  };
}