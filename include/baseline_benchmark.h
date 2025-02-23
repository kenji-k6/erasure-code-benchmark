#ifndef BASELINE_BENCHMARK_H
#define BASELINE_BENCHMARK_H

#include "abstract_benchmark.h"
#include "baseline_ecc.h"
#include "utils.h"


#include <cstdint>
#include <cstddef>
#include <iostream>
#include <vector>
#include <memory>
#include <cstring>


class BaselineBenchmark : public ECCBenchmark {
public:
  int setup() noexcept override;
  void teardown() noexcept override;
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
  bool check_for_corruption() const noexcept override;
  void flush_cache() noexcept override;
  

private:
  Encoder encoder_;
  Decoder decoder_;

  uint32_t *inv_mat;
  uint8_t *orig_data;
  uint8_t *redundant_data;
  uint32_t recv_idx[256];
}; // class BaselineBenchmark


#endif // BASELINE_BENCHMARK_H