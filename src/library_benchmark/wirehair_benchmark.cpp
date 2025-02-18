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

  decoded_buffer_ = (uint8_t*) simd_safe_allocate(kConfig.data_size);
  if (!decoded_buffer_) {
    teardown();
    std::cerr << "Wirehair: Failed to allocate memory for decoded data.\n";
    return -1;
  }

  // Create the encoder
  encoder_ = wirehair_encoder_create(nullptr, original_buffer_, kConfig.data_size, kConfig.block_size);
  if (!encoder_) {
    teardown();
    std::cerr << "Wirehair: Failed to create encoder.\n";
    return -1;
  }

  // Create the decoder
  decoder_ = wirehair_decoder_create(nullptr, kConfig.data_size, kConfig.block_size);
  if (!decoder_) {
    teardown();
    std::cerr << "Wirehair: Failed to create decoder.\n";
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

  return 0;
}



void WirehairBenchmark::teardown() {
  if (original_buffer_) simd_safe_free(original_buffer_);
  if (encoded_buffer_) simd_safe_free(encoded_buffer_);
  if (encoder_) wirehair_free(encoder_);
  if (decoder_) wirehair_free(decoder_);
}



int WirehairBenchmark::encode() {
  unsigned i = 0, block_id = 0, needed = 0;
  uint32_t write_len = 0;

  for (;;) {
    block_id++;
    needed++;

    // Encode the block
    WirehairResult encode_result = wirehair_encode(
      encoder_,
      block_id,
      encoded_buffer_ + i,
      kConfig.block_size,
      &write_len
    );

    if (/*needed > kConfig.computed.recovery_blocks ||*/ encode_result != Wirehair_Success) {
      teardown();
      std::cerr << "Wirehair: Failed to encode data, too many recovery blocks needed. (" << wirehair_result_string(encode_result) << ")\n";
      return -1;
    }

    WirehairResult decode_result = wirehair_decode(
      decoder_,
      block_id,
      encoded_buffer_ + i,
      write_len
    );

    i += kConfig.block_size;

    if (decode_result == Wirehair_Success) {
      break;
    }

    if (decode_result != Wirehair_NeedMore) {
      teardown();
      std::cerr << "Wirehair: Failed to encode data. \n";
      return -1;
    }
  }
  return 0;
}



int WirehairBenchmark::decode() {
  return wirehair_recover(
    decoder_,
    decoded_buffer_,
    kConfig.data_size
  );
}



void WirehairBenchmark::flush_cache() {
  // TODO: Implement cache flushing
}



bool WirehairBenchmark::check_for_corruption() {
  return true;
}



void WirehairBenchmark::simulate_data_loss() {
  // TODO: Implement data loss simulation
}