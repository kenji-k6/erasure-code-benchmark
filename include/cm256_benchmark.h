#ifndef CM256_BENCHMARK_H
#define CM256_BENCHMARK_H

#include "benchmark.h"
#include "cm256.h"

#include <iostream>
#include <chrono>
#include <vector>
#include <memory>
#include <cstddef>
#include <cstring>

#define CM256_MIN_BLOCKS 1
#define CM256_MAX_BLOCKS 256

class CM256Benchmark : public ECCBenchmark {
public:
  int setup(const BenchmarkConfig& config) override;
  int encode() override;
  int decode(double loss_rate) override;
  void teardown() override;
  Metrics get_metrics() const override;

private:
  cm256_encoder_params params_; // Encoder parameters
  uint8_t* original_data_;
  uint8_t* recovery_data_;
  cm256_block blocks_[256];

  long long encode_time_us_ = 0;
  long long decode_time_us_ = 0;
  double encode_input_throughput_mbps_ = 0.0;
  double encode_output_throughput_mbps_ = 0.0;
  double decode_input_throughput_mbps_ = 0.0;
  double decode_output_throughput_mbps_ = 0.0;
  BenchmarkConfig config_;
};

#endif // CM256_BENCHMARK_H