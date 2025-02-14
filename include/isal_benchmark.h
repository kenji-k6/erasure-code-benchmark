#ifndef ISAL_BENCHMARK_H
#define ISAL_BENCHMARK_H

#include "abstract_benchmark.h"
#include "erasure_code.h"
#include "utils.h"

#include <iostream>
#include <vector>
#include <memory>
#include <cstring>

class ISALBenchmark : public ECCBenchmark {
public:
  int setup() override;
  void teardown() override;
  int encode() override;
  int decode() override;
  void flush_cache() override;
  void check_for_corruption() override;
  void simulate_data_loss() override;
  

private:
  uint8_t* original_data_;
  uint8_t* recovery_data_;
  uint8_t** original_data_ptrs_;
  uint8_t** recovery_data_ptrs_;
  uint8_t* encode_matrix_;
  uint8_t* decode_matrix_;
  uint8_t* table_;
}; // class ISALBenchmark
#endif // ISAL_BENCHMARK_H