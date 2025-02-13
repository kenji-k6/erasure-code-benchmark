#ifndef LEOPARD_BENCHMARK_H
#define LEOPARD_BENCHMARK_H

#include "benchmark.h"
#include "leopard.h"

#include <iostream>
#include <chrono>
#include <vector>
#include <memory>
#include <cstddef>
#include <cstring>

#define LEOPARD_MIN_BLOCKS 2
#define LEOPARD_MAX_BLOCKS 65536
#define LEOPARD_BLOCK_SIZE_ALIGNMENT 64

class LeopardBenchmark : public ECCBenchmark {
public:
  int setup(const BenchmarkConfig& config) override;
  int encode() override;
  int decode(double loss_rate) override;
  void teardown() override;
  Metrics get_metrics() const override;

private:
  unsigned encode_work_count_ = 0;
  unsigned decode_work_count_ = 0;
  std::vector<void*> original_ptrs_;
  std::vector<void*> encode_work_ptrs_;
  std::vector<void*> decode_work_ptrs_;

  long long encode_time_us_ = 0;
  long long decode_time_us_ = 0;
  double encode_input_throughput_mbps_ = 0.0;
  double encode_output_throughput_mbps_ = 0.0;
  double decode_input_throughput_mbps_ = 0.0;
  double decode_output_throughput_mbps_ = 0.0;
  BenchmarkConfig config_;
};


#endif // LEOPARD_BENCHMARK_H