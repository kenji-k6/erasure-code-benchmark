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

  // Initialize original data to 1s, work data to 0s
  memset(original_buffer_, 0xFF, kConfig.data_size);
  memset(encode_work_buffer_, 0xF1, kConfig.block_size * encode_work_count_);
  memset(decode_work_buffer_, 0xF1, kConfig.block_size * decode_work_count_);

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


  // Set certain data for later checking:
  for (unsigned i = 0; i < kConfig.block_size; i++) {
    original_ptrs_[5][i] = i%256;
    original_ptrs_[6][i] = i%256;
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
  // TODO: Implement corruption checking
 
  for (unsigned i = 0; i < kConfig.computed.original_blocks; i++) {
    for (unsigned j = 0; j < kConfig.block_size; j++) {
      if (i == 5 || i == 6) {
        if (decode_work_ptrs_[i][j] != j%256) {
          std::cout << "Corruption detected at block " << i << ", byte " << j << "( " << (int)decode_work_ptrs_[i][j] << " != " << j << ")\n";
          return true;
        }
      } else {
        if (original_ptrs_[i][j] != 0xFF) {
          return true;
        }
      }
    }
  }

  return false;
}



void LeopardBenchmark::simulate_data_loss() {
  // TODO: Implement data loss simulation
  
  for (unsigned i = 5; i < 7; i++) {
    // Simulate data loss
    memset(&original_buffer_[i*kConfig.block_size], 0, kConfig.block_size);
    original_ptrs_[i] = nullptr;
  }

}