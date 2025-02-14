#ifndef LEOPARD_BENCHMARK_H
#define LEOPARD_BENCHMARK_H

#include "abstract_benchmark.h"
#include "leopard.h"
#include "utils.h"

#include <iostream>
#include <vector>
#include <memory>
#include <cstring>


class LeopardBenchmark : public ECCBenchmark {
public:
  int setup(const BenchmarkConfig& config) override;
  void teardown() override;
  int encode() override;
  int decode() override;
  void flush_cache() override;
  void check_for_corruption() override;
  void simulate_data_loss() override;
  

private:
  unsigned encode_work_count_ = 0;
  unsigned decode_work_count_ = 0;
  void* original_buffer_; 
  void* encode_work_buffer_;
  void* decode_work_buffer_;
  void** original_ptrs_;
  void** encode_work_ptrs_;
  void** decode_work_ptrs_;
  BenchmarkConfig config_;
};

#endif // LEOPARD_BENCHMARK_H