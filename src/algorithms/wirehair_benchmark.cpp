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


WirehairBenchmark::WirehairBenchmark(const BenchmarkConfig& config) noexcept : ECBenchmark(config) {
  num_total_blocks_ = num_original_blocks_ + num_recovery_blocks_;
  if (wirehair_init()) throw_error("Wirehair: Initialization failed.");

  // Allocate buffers
  original_buffer_ = std::make_unique<uint8_t[]>(block_size_ * num_original_blocks_);
  encode_buffer_ = std::make_unique<uint8_t[]>(block_size_ * num_total_blocks_);
  decode_buffer_ = std::make_unique<uint8_t[]>(block_size_ * num_original_blocks_);

  if (!original_buffer_ || !encode_buffer_ || !decode_buffer_) throw_error("Wirehair: Failed to allocate buffer(s).");

  decoder_ = wirehair_decoder_create(nullptr, block_size_ * num_original_blocks_, block_size_);
  if (!decoder_) throw_error("Wirehair: Failed to create decoder instance.");

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < num_original_blocks_; ++i) {
    int write_res = write_validation_pattern(i, &original_buffer_[i * block_size_], block_size_);
    if (write_res) throw_error("Wirehair: Failed to write random checking packet.");
  }
}

WirehairBenchmark::~WirehairBenchmark() noexcept {
  if (encoder_) wirehair_free(encoder_);
  if (decoder_) wirehair_free(decoder_);
}

int WirehairBenchmark::encode() noexcept {
  encoder_ = wirehair_encoder_create(nullptr, original_buffer_.get(), num_original_blocks_ * block_size_, block_size_);
  if (!encoder_) return -1;
  uint32_t write_len = 0;
  for (size_t i = 0; i < num_original_blocks_ + num_recovery_blocks_; i++) {
    if (wirehair_encode(encoder_, i, &encode_buffer_[i * block_size_],
                        block_size_, &write_len) != Wirehair_Success) return -1;
  }
  return 0;
}


int WirehairBenchmark::decode() noexcept {
  WirehairResult decode_result = Wirehair_NeedMore;
  unsigned loss_idx = 0;

  for (unsigned i = 0; i < num_total_blocks_; i++) {
    if (loss_idx < num_lost_blocks_ && i == lost_block_idxs_[loss_idx]) {
      loss_idx++;
      continue;
    }

    decode_result = wirehair_decode(decoder_, i, &encode_buffer_[i * block_size_], block_size_);
    if (decode_result == Wirehair_Success) break;
  }

  return wirehair_recover(decoder_, decode_buffer_.get(), block_size_ * num_original_blocks_);
}


void WirehairBenchmark::simulate_data_loss() noexcept {
  for (auto idx : lost_block_idxs_) {
    memset(&encode_buffer_[idx * block_size_], 0, block_size_);
  }
}


bool WirehairBenchmark::check_for_corruption() const noexcept {
  for (unsigned i = 0; i < num_original_blocks_; i++) {
    if (!validate_block(&decode_buffer_[i * block_size_], block_size_)) return false;
  }
  return true;
}