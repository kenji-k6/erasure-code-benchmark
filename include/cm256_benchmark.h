#ifndef CM256_BENCHMARK_H
#define CM256_BENCHMARK_H

#include "abstract_benchmark.h"
#include "cm256.h"
#include "utils.h"
#include "benchmark/benchmark.h"

#include <iostream>
#include <vector>
#include <memory>
#include <cstring>


class CM256Benchmark : public ECCBenchmark {
public:
  int setup() override;
  void teardown() override;
  int encode() override;
  int decode() override;
  void flush_cache() override;
  void check_for_corruption() override;
  void simulate_data_loss() override;

private:
  cm256_encoder_params params_; // Encoder parameters
  uint8_t* original_buffer_;
  uint8_t* recovery_buffer_;
  cm256_block blocks_[256];
};

#endif // CM256_BENCHMARK_H