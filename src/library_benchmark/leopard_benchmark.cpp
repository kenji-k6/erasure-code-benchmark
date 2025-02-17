#include "leopard_benchmark.h"

int LeopardBenchmark::setup() {
  // Initialize Leopard
  if (leo_init()) {
    std::cerr << "Leopard: Initialization failed.\n";
    return -1;
  }

  // Compute encode/decode work counts
  encode_work_count_ = leo_encode_work_count(kConfig.computed.original_blocks, kConfig.computed.recovery_blocks);
  decode_work_count_ = leo_decode_work_count(kConfig.computed.original_blocks, kConfig.computed.recovery_blocks);

  if (encode_work_count_ == 0 || decode_work_count_ == 0) {
    std::cerr << "Leopard: Invalid work count(s): encode=" << encode_work_count_ << ", decode=" << decode_work_count_ << "\n";
    return -1;
  }

  // Allocate buffers
  original_buffer_ = (uint8_t*) simd_safe_allocate(kConfig.block_size * kConfig.computed.original_blocks);
  if (!original_buffer_) {
    teardown();
    std::cerr << "Leopard: Failed to allocate original buffer.\n";
    return -1;
  }

  encode_work_buffer_ = (uint8_t*) simd_safe_allocate(kConfig.block_size * encode_work_count_);
  if (!encode_work_buffer_) {
    teardown();
    std::cerr << "Leopard: Failed to allocate encode work buffer.\n";
    return -1;
  }

  decode_work_buffer_ = (uint8_t*) simd_safe_allocate(kConfig.block_size * decode_work_count_);
  if (!decode_work_buffer_) {
    teardown();
    std::cerr << "Leopard: Failed to allocate decode work buffer.\n";
    return -1;
  }

  // Allocate pointers
  original_ptrs_ = new uint8_t* [kConfig.computed.original_blocks];
  encode_work_ptrs_ = new uint8_t*[encode_work_count_];
  decode_work_ptrs_ = new uint8_t*[decode_work_count_];

  if (!original_ptrs_ || !encode_work_ptrs_ || !decode_work_ptrs_) {
    teardown();
    std::cerr << "Leopard: Failed to allocate pointer arrays.\n";
    return -1;
  }

  // Initialize pointers
  for (unsigned i = 0; i < kConfig.computed.original_blocks; i++) {
    original_ptrs_[i] = (uint8_t*) (((uint8_t*)original_buffer_) + i * kConfig.block_size);
  }

  for (unsigned i = 0; i < encode_work_count_; i++) {
    encode_work_ptrs_[i] = (uint8_t*) (((uint8_t*)encode_work_buffer_) + i * kConfig.block_size);
  }

  for (unsigned i = 0; i < decode_work_count_; i++) {
    decode_work_ptrs_[i] = (uint8_t*) (((uint8_t*)decode_work_buffer_) + i * kConfig.block_size);
  }


  // Initialize original data to random values
  for (unsigned i = 0; i < kConfig.computed.original_blocks; i++) {
    set_block_check_values(
      original_ptrs_[i],
      kConfig.block_size,
      i
    );
  }

  return 0;
}



void LeopardBenchmark::teardown() {
  if (original_buffer_) simd_safe_free(original_buffer_);
  if (encode_work_buffer_) simd_safe_free(encode_work_buffer_);
  if (decode_work_buffer_) simd_safe_free(decode_work_buffer_);
  if (original_ptrs_) delete[] original_ptrs_;
  if (encode_work_ptrs_) delete[] encode_work_ptrs_;
  if (decode_work_ptrs_) delete[] decode_work_ptrs_;
}



int LeopardBenchmark::encode() {
  // Encode the data
  return leo_encode(
    kConfig.block_size,
    kConfig.computed.original_blocks,
    kConfig.computed.recovery_blocks,
    encode_work_count_,
    (void**)original_ptrs_,
    (void**)encode_work_ptrs_
  );
}



int LeopardBenchmark::decode() {
  // Decode the data
  return leo_decode(
    kConfig.block_size,
    kConfig.computed.original_blocks,
    kConfig.computed.recovery_blocks,
    decode_work_count_,
    (void**)original_ptrs_,
    (void**)encode_work_ptrs_,
    (void**)decode_work_ptrs_
  );
}



void LeopardBenchmark::flush_cache() {
  // TODO: Implement cache flushing
}



bool LeopardBenchmark::check_for_corruption() {
  for (unsigned i = 0; i < kConfig.computed.original_blocks; i++) {
    if (i < kConfig.computed.num_lost_data_blocks) {
      if (check_block_for_corruption(decode_work_ptrs_[i], kConfig.block_size, i)) {
        return true;
      }

    } else {
      if (check_block_for_corruption(original_ptrs_[i], kConfig.block_size, i)) {
        return true;
      }
    }
  }
  return false;
}



void LeopardBenchmark::simulate_data_loss() {
  // TODO: Allow for general lost block indices, right now we always start at block 0
  for (unsigned i = 0; i < kConfig.computed.num_lost_data_blocks; i++) {
    // Simulate data loss
    original_ptrs_[i] = nullptr;
  }

  for (unsigned i = 0; i < kConfig.computed.num_lost_recovery_blocks; i++) {
    // Simulate recovery block loss
    encode_work_ptrs_[i] = nullptr;
  }
}