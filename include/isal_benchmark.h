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
  // Fragment / block buffer pointers
  uint8_t* original_ptrs_[ISAL_MAX_TOT_BLOCKS];
  uint8_t* recovery_src_ptrs_[ISAL_MAX_ORIG_BLOCKS];
  uint8_t* recovery_dest_ptrs_[ISAL_MAX_TOT_BLOCKS];
  uint8_t block_err_list[ISAL_MAX_TOT_BLOCKS];

  // Coefficient matrices
  uint8_t* encode_matrix_;
  uint8_t* decode_matrix_;
  uint8_t* invert_matrix_;
  uint8_t* temp_matrix;
  uint8_t* g_tblts_;
  uint8_t decode_index[ISAL_MAX_TOT_BLOCKS];
}; // class ISALBenchmark
#endif // ISAL_BENCHMARK_H