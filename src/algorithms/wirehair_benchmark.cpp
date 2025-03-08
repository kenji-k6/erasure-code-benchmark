/**
 * @file wirehair_benchmark.cpp
 * @brief Benchmark implementation for the Wirehair EC library
 * 
 * Documentation can be found in wirehair_benchmark.h and abstract_benchmark.h
 */


#include "wirehair_benchmark.h"

#include "utils.h"
#include <cstring>
#include <iostream>


WirehairBenchmark::WirehairBenchmark(const BenchmarkConfig& config) noexcept : ECBenchmark(config) {}

int WirehairBenchmark::setup() noexcept {

  // Initialize Wirehair
  if (wirehair_init()) {
    std::cerr << "Wirehair: Initialization failed.\n";
    return -1;
  }

  // Allocate buffer with proper alignment for SIMD
  original_buffer_ = static_cast<uint8_t*>(aligned_alloc(ALIGNMENT_BYTES, block_size_ * num_original_blocks_));
  encode_buffer_ = static_cast<uint8_t*>(aligned_alloc(ALIGNMENT_BYTES, block_size_ * (num_original_blocks_ + num_recovery_blocks_)));
  decode_buffer_ = static_cast<uint8_t*>(aligned_alloc(ALIGNMENT_BYTES, block_size_ * num_original_blocks_));

  if (!original_buffer_ || !encode_buffer_ || !decode_buffer_) {
    std::cerr << "Wirehair: Failed to allocate buffer(s).\n";
    teardown();
    return -1;
  }

  // Create the decoder instance
  decoder_ = wirehair_decoder_create(nullptr, block_size_ * num_original_blocks_, block_size_);

  if (!decoder_) {
    std::cerr << "Wirehair: Failed to create decoder instance.\n";
    teardown();
    return -1;
  }

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < num_original_blocks_; i++) {
    int write_res = write_validation_pattern(i, original_buffer_ + i * block_size_, block_size_);
    if (write_res) {
      std::cerr << "Wirehair: Failed to write random checking packet.\n";
      teardown();
      return -1;
    }
  }

  return 0;
}


void WirehairBenchmark::teardown() noexcept {
  if (original_buffer_) free(original_buffer_);
  if (encode_buffer_) free(encode_buffer_);
  if (decode_buffer_) free(decode_buffer_);
  if (encoder_) wirehair_free(encoder_);
  if (decoder_) wirehair_free(decoder_);
}


int WirehairBenchmark::encode() noexcept {
  encoder_ = wirehair_encoder_create(nullptr, original_buffer_, num_original_blocks_ * block_size_, block_size_);
  if (!encoder_) return -1;

  uint32_t write_len = 0;
  for (size_t i = 0; i < num_original_blocks_ + num_recovery_blocks_; i++) {
    if (wirehair_encode(encoder_, i, encode_buffer_ + i * block_size_,
                        block_size_, &write_len) != Wirehair_Success) return -1;
  }
  return 0;
}


int WirehairBenchmark::decode() noexcept {
  WirehairResult decode_result = Wirehair_NeedMore;
  unsigned loss_idx = 0;

  for (unsigned i = 0; i < num_original_blocks_ + num_recovery_blocks_; i++) {
    if (loss_idx < num_lost_blocks_ && i == lost_block_idxs_[loss_idx]) {
      loss_idx++;
      continue;
    }

    decode_result = wirehair_decode(decoder_, i, encode_buffer_ + i * block_size_, block_size_);
    if (decode_result == Wirehair_Success) break;
  }

  return wirehair_recover(decoder_, decode_buffer_, block_size_ * num_original_blocks_);
}


void WirehairBenchmark::simulate_data_loss() noexcept {
  for (unsigned i = 0; i < num_lost_blocks_; i++) {
    memset(encode_buffer_ + (lost_block_idxs_[i] * block_size_), 0, block_size_);
  }
}


bool WirehairBenchmark::check_for_corruption() const noexcept {
  for (unsigned i = 0; i <num_original_blocks_; i++) {
    if (!validate_block(decode_buffer_ + i * block_size_, block_size_)) return false;
  }
  return true;
}

void LeopardBenchmark::invalidate_memory() noexcept {}
