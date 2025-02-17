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
  int setup() override;
  void teardown() override;
  int encode() override;
  int decode() override;
  void flush_cache() override;
  bool check_for_corruption() override;
  void simulate_data_loss() override;
  

private:
  unsigned encode_work_count_ = 0;
  unsigned decode_work_count_ = 0;
  uint8_t* original_buffer_; 
  uint8_t* encode_work_buffer_;
  uint8_t* decode_work_buffer_;
  uint8_t** original_ptrs_;
  uint8_t** encode_work_ptrs_;
  uint8_t** decode_work_ptrs_;
}; // class LeopardBenchmark

#endif // LEOPARD_BENCHMARK_H