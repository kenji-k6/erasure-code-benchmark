#include "wirehair_benchmark.h"

int WirehairBenchmark::setup(const BenchmarkConfig& config) {
  // TODO: Implement

  config_ = config;

  // 64 byte alignment not needed for wirehair
  if (config_.computed.original_blocks < WIREHAIR_MIN_BLOCKS || config_.computed.original_blocks > WIREHAIR_MAX_BLOCKS) {
    std::cerr << "Wirehair: Original blocks must be between " << WIREHAIR_MIN_BLOCKS << " and " << WIREHAIR_MAX_BLOCKS << " (is " << config_.computed.original_blocks << ").\n";
    return -1;
  }

  // Original data
  original_data_.resize(config_.data_size);
  
  // Fill with constant 1 data
  memset(&original_data_[0], 1, config_.data_size);

  // Initialize the encoder
  encoder_ = wirehair_encoder_create(nullptr, &original_data_[0], config.data_size, config.block_size);
  if (!encoder_) {
    std::cerr << "Wirehair: Encoder initialization failed.\n";
    return -1;
  }

  // Initialize the decoder
  decoder_ = wirehair_decoder_create(nullptr, config.data_size, config.block_size);
  if (!decoder_) {
    wirehair_free(encoder_);
    std::cerr << "Wirehair: Decoder initialization failed.\n";
    return -1;
  }

  return 0;
}


int WirehairBenchmark::encode() {
  // TODO: Implement
  return -1;
}


int WirehairBenchmark::decode(double loss_rate) {
  // TODO: Implement
  return -1;
}


void WirehairBenchmark::teardown() {
  // TODO: Implement
  return;
}


ECCBenchmark::Metrics WirehairBenchmark::get_metrics() const {
  // TODO: Implement
  return {0,0,0,0,0,0};
}