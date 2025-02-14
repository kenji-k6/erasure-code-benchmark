#include "wirehair_benchmark.h"

int WirehairBenchmark::setup() {
  // Initialize Wirehair
  if (wirehair_init()) {
    std::cerr << "Wirehair: Initialization failed.\n";
    return -1; 
  }

  // Allocate memory for the original data
  original_data_ = (uint8_t*) simd_safe_allocate(kConfig.data_size);
  if (!original_data_) {
    teardown();
    std::cerr << "Wirehair: Failed to allocate memory for original data.\n";
    return -1;
  }

  // Initialize data with 1s
  memset(original_data_, 1, kConfig.data_size);

  // Create the encoder
  encoder_ = wirehair_encoder_create(nullptr, original_data_, kConfig.data_size, kConfig.block_size);
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

  return 0;
}

void WirehairBenchmark::teardown() {
  if (original_data_) simd_safe_free(original_data_);
  if (encoder_) wirehair_free(encoder_);
  if (decoder_) wirehair_free(decoder_);
}