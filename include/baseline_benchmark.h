#ifndef BASELINE_BENCHMARK_H
#define BASELINE_BENCHMARK_H

#include "abstract_benchmark.h"
#include "utils.h"
#include "baseline.h"


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
  Baseline_Params params_;
  uint8_t *InvMatPtr_;
}; // class BaselineBenchmark


#endif // BASELINE_BENCHMARK_H