#ifndef WIREHAIR_BENCHMARK_H
#define WIREHAIR_BENCHMARK_H
#include "abstract_benchmark.h"
#include "wirehair/wirehair.h"
#include "utils.h"

#include <iostream>
#include <vector>
#include <memory>
#include <cstring>

class WirehairBenchmark : public ECCBenchmark {
public:
  int setup() noexcept override;
  void teardown() noexcept override;
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
  bool check_for_corruption() const noexcept override;
  void flush_cache() noexcept override;

private:
  uint8_t* original_buffer_;
  uint8_t* encoded_buffer_;
  uint8_t* decoded_buffer_;
  WirehairCodec encoder_;
  WirehairCodec decoder_;
}; // class WirehairBenchmark

#endif // WIREHAIR_BENCHMARK_H
