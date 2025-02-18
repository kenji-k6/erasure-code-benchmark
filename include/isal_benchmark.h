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
  int setup() noexcept override;
  void teardown() noexcept override;
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
  bool check_for_corruption() const noexcept override;
  void flush_cache() noexcept override;
  

private:
  // Buffers
  uint8_t* original_data_;
  uint8_t* recovery_outp_data_;

  // Fragment / block buffer pointers
  uint8_t* original_ptrs_[ECCLimits::ISAL_MAX_TOT_BLOCKS];
  uint8_t* recovery_src_ptrs_[ECCLimits::ISAL_MAX_DATA_BLOCKS];
  uint8_t* recovery_outp_ptrs_[ECCLimits::ISAL_MAX_DATA_BLOCKS];
  uint8_t block_err_list_[ECCLimits::ISAL_MAX_TOT_BLOCKS];


  // Coefficient matrices
  uint8_t* encode_matrix_;
  uint8_t* decode_matrix_;
  uint8_t* invert_matrix_;
  uint8_t* temp_matrix_;
  uint8_t* g_tbls_;
  uint8_t decode_index[ECCLimits::ISAL_MAX_TOT_BLOCKS];
}; // class ISALBenchmark


static int gf_gen_decode_matrix_simple(uint8_t *encode_matrix, uint8_t *decode_matrix, uint8_t *invert_matrix, uint8_t *temp_matrix, uint8_t *decode_index, uint8_t *frag_err_list, int nerrs, int k, int m);
#endif // ISAL_BENCHMARK_H