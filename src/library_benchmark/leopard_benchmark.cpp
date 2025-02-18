#include "leopard_benchmark.h"

int LeopardBenchmark::setup() noexcept {
  // Initialize Leopard
  if (leo_init()) {
    std::cerr << "Leopard: Initialization failed.\n";
    return -1;
  }

  // Compute encode/decode work counts
  encode_work_count_ = leo_encode_work_count(benchmark_config.computed.num_original_blocks, benchmark_config.computed.num_recovery_blocks);
  decode_work_count_ = leo_decode_work_count(benchmark_config.computed.num_original_blocks, benchmark_config.computed.num_recovery_blocks);

  if (encode_work_count_ == 0 || decode_work_count_ == 0) {
    std::cerr << "Leopard: Invalid work count(s): encode=" << encode_work_count_ << ", decode=" << decode_work_count_ << "\n";
    return -1;
  }

  // Allocate buffers
  original_buffer_ = (uint8_t*) aligned_alloc(ALIGNMENT_BYTES, benchmark_config.block_size * benchmark_config.computed.num_original_blocks);
  if (!original_buffer_) {
    teardown();
    std::cerr << "Leopard: Failed to allocate original buffer.\n";
    return -1;
  }

  encode_work_buffer_ = (uint8_t*) aligned_alloc(ALIGNMENT_BYTES, benchmark_config.block_size * encode_work_count_);
  if (!encode_work_buffer_) {
    teardown();
    std::cerr << "Leopard: Failed to allocate encode work buffer.\n";
    return -1;
  }

  decode_work_buffer_ = (uint8_t*) aligned_alloc(ALIGNMENT_BYTES, benchmark_config.block_size * decode_work_count_);
  if (!decode_work_buffer_) {
    teardown();
    std::cerr << "Leopard: Failed to allocate decode work buffer.\n";
    return -1;
  }

  // Allocate pointers
  original_ptrs_ = new uint8_t* [benchmark_config.computed.num_original_blocks];
  encode_work_ptrs_ = new uint8_t*[encode_work_count_];
  decode_work_ptrs_ = new uint8_t*[decode_work_count_];

  if (!original_ptrs_ || !encode_work_ptrs_ || !decode_work_ptrs_) {
    teardown();
    std::cerr << "Leopard: Failed to allocate pointer arrays.\n";
    return -1;
  }

  // Initialize pointers
  for (unsigned i = 0; i < benchmark_config.computed.num_original_blocks; i++) {
    original_ptrs_[i] = (uint8_t*) (((uint8_t*)original_buffer_) + i * benchmark_config.block_size);
  }

  for (unsigned i = 0; i < encode_work_count_; i++) {
    encode_work_ptrs_[i] = (uint8_t*) (((uint8_t*)encode_work_buffer_) + i * benchmark_config.block_size);
  }

  for (unsigned i = 0; i < decode_work_count_; i++) {
    decode_work_ptrs_[i] = (uint8_t*) (((uint8_t*)decode_work_buffer_) + i * benchmark_config.block_size);
  }


  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < benchmark_config.computed.num_original_blocks; i++) {
    int write_res = write_validation_pattern(
      i,
      original_ptrs_[i],
      benchmark_config.block_size
    );

    if (write_res) {
      teardown();
      std::cerr << "Leopard: Failed to write random checking packet.\n";
      return -1;
    }
  }

  return 0;
}



void LeopardBenchmark::teardown() noexcept {
  if (original_buffer_) free(original_buffer_);
  if (encode_work_buffer_) free(encode_work_buffer_);
  if (decode_work_buffer_) free(decode_work_buffer_);
  if (original_ptrs_) delete[] original_ptrs_;
  if (encode_work_ptrs_) delete[] encode_work_ptrs_;
  if (decode_work_ptrs_) delete[] decode_work_ptrs_;
}



int LeopardBenchmark::encode() noexcept {
  // Encode the data
  return leo_encode(
    benchmark_config.block_size,
    benchmark_config.computed.num_original_blocks,
    benchmark_config.computed.num_recovery_blocks,
    encode_work_count_,
    (void**)original_ptrs_,
    (void**)encode_work_ptrs_
  );
}



int LeopardBenchmark::decode() noexcept {
  // Decode the data
  return leo_decode(
    benchmark_config.block_size,
    benchmark_config.computed.num_original_blocks,
    benchmark_config.computed.num_recovery_blocks,
    decode_work_count_,
    (void**)original_ptrs_,
    (void**)encode_work_ptrs_,
    (void**)decode_work_ptrs_
  );
}



void LeopardBenchmark::flush_cache() noexcept {
  // TODO: Implement cache flushing
}



bool LeopardBenchmark::check_for_corruption() const noexcept {
  for (unsigned i = 0; i < benchmark_config.computed.num_original_blocks; i++) {
    bool res = false;
    if (!original_ptrs_[i]) { // lost block
      res = validate_block(decode_work_ptrs_[i], benchmark_config.block_size);
    } else { // did not lose block
      res = validate_block(original_ptrs_[i], benchmark_config.block_size);
    }

    if (!res) return false;
  }
  return true;
}



void LeopardBenchmark::simulate_data_loss() noexcept {
  // Lost block indices are already computed
  // They can be found in lost_block_idxs

  for (unsigned i = 0; i < benchmark_config.num_lost_blocks; i++) {
    uint32_t idx = lost_block_idxs[i];
    // if idx >= benchmark_config.computed.num_original_blocks, then it is a recovery block
    // else it's a data block

    if (idx < benchmark_config.computed.num_original_blocks) {
      // Zero out the block
      memset(original_ptrs_[idx], 0, benchmark_config.block_size);
      // Set corresponding block ptr to a nullptr
      original_ptrs_[idx] = nullptr;
    } else {
      idx -= benchmark_config.computed.num_original_blocks;
      // Zero out the block
      memset(encode_work_ptrs_[idx], 0, benchmark_config.block_size);
      // Set corresponding block ptr to a nullptr
      encode_work_ptrs_[idx] = nullptr;
    }
  }
}