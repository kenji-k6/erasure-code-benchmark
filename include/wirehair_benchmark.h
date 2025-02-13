#ifndef WIREHAIR_BENCHMARK_H
#define WIREHAIR_BENCHMARK_H

#include "benchmark.h"
#include "wirehair/wirehair.h"

#include <iostream>
#include <chrono>
#include <vector>
#include <memory>
#include <cstddef>
#include <cstring>

#define WIREHAIR_MIN_BLOCKS 2
#define WIREHAIR_MAX_BLOCKS 64000


class WirehairBenchmark : public ECCBenchmark {
public:
  int setup(const BenchmarkConfig& config) override;
  int encode() override;
  int decode(double loss_rate) override;
  void teardown() override;
  Metrics get_metrics() const override;


private:
  long long encode_time_us_ = 0;
  long long decode_time_us_ = 0;
  double encode_input_throughput_mbps_ = 0.0;
  double encode_output_throughput_mbps_ = 0.0;
  double decode_input_throughput_mbps_ = 0.0;
  double decode_output_throughput_mbps_ = 0.0;
  size_t memory_used_ = 0;
  size_t total_data_bytes_ = 0;
  BenchmarkConfig config_;
}; // class WirehairBenchmark

#endif // WIREHAIR_BENCHMARK_H