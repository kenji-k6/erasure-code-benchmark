#include "wirehair_benchmark.h"

int WirehairBenchmark::setup() {
  // Initialize Wirehair
  if (wirehair_init()) {
    std::cerr << "Wirehair: Initialization failed.\n";
    return -1; 
  }

  // Allocate memory for the buffers
  original_buffer_ = (uint8_t*) simd_safe_allocate(kConfig.block_size * kConfig.computed.original_blocks);
  if (!original_buffer_) {
    teardown();
    std::cerr << "Wirehair: Failed to allocate memory for original data.\n";
    return -1;
  }

  encoded_buffer_ = (uint8_t*) simd_safe_allocate(kConfig.block_size * (kConfig.computed.original_blocks+kConfig.computed.recovery_blocks));
  if (!encoded_buffer_) {
    teardown();
    std::cerr << "Wirehair: Failed to allocate memory for output data.\n";
    return -1;
  }

  decoded_buffer_ = (uint8_t*) simd_safe_allocate(kConfig.block_size * kConfig.computed.original_blocks);
  if (!decoded_buffer_) {
    teardown();
    std::cerr << "Wirehair: Failed to allocate memory for decoded data.\n";
    return -1;
  }

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < kConfig.computed.original_blocks; i++) {
    int write_res = write_random_checking_packet(
      i,
      original_buffer_ + i * kConfig.block_size,
      kConfig.block_size
    );

    if (write_res) {
      teardown();
      std::cerr << "Wirehair: Failed to write random checking packet.\n";
      return -1;
    }
  }


  // Create the decoder
  decoder_ = wirehair_decoder_create(
    nullptr,
    kConfig.data_size,
    kConfig.block_size
  );
  if (!decoder_) return -1;

  return 0;
}



void WirehairBenchmark::teardown() {
  if (original_buffer_) simd_safe_free(original_buffer_);
  if (encoded_buffer_) simd_safe_free(encoded_buffer_);
  if (encoder_) wirehair_free(encoder_);
  if (decoder_) wirehair_free(decoder_);
}



int WirehairBenchmark::encode() {
  
  uint32_t write_len = 0;
  WirehairResult encode_result;
  // Create the encoder
  encoder_ = wirehair_encoder_create(
    nullptr,
    original_buffer_,
    kConfig.data_size,
    kConfig.block_size
  );
  if (!encoder_) return -1;

  for (unsigned i = 0; i < kConfig.computed.original_blocks + kConfig.computed.recovery_blocks; i++) {

    encode_result = wirehair_encode(
      encoder_,
      i,
      encoded_buffer_ + (i * kConfig.block_size),
      kConfig.block_size,
      &write_len
    );

    if (encode_result != Wirehair_Success) return -1;
  }
  return 0;
}



int WirehairBenchmark::decode() {
  WirehairResult decode_result = Wirehair_NeedMore;
  unsigned loss_idx = 0;

  for (unsigned i = 0; i < kConfig.computed.original_blocks + kConfig.computed.recovery_blocks; i++) {

    if (i == kLost_block_idxs[loss_idx]) {
      loss_idx++;
      continue;
    }

    decode_result = wirehair_decode(
      decoder_,
      i,
      encoded_buffer_ + (i * kConfig.block_size),
      kConfig.block_size
    );

    if (decode_result == Wirehair_Success) break;
  }

  return wirehair_recover(
    decoder_,
    decoded_buffer_,
    kConfig.computed.original_blocks * kConfig.block_size
  );
}



void WirehairBenchmark::flush_cache() {
  // TODO: Implement cache flushing
}



bool WirehairBenchmark::check_for_corruption() {
  for (unsigned i = 0; i < kConfig.computed.original_blocks; i++) {
    if (!check_packet(decoded_buffer_ + (i * kConfig.block_size), kConfig.block_size)) {
      return false;
    }
  }
  return true;
}


void WirehairBenchmark::simulate_data_loss() {
  /* Loss logic is also part of decode function!!! */
  for (unsigned i = 0; i < kConfig.num_lost_blocks; i++) {
    uint32_t idx = kLost_block_idxs[i];
    memset(encoded_buffer_ + (idx * kConfig.block_size), 0, kConfig.block_size);
  }
}