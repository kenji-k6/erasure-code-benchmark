#ifndef BASELINE_BENCHMARK_H
#define BASELINE_BENCHMARK_H

#include "abstract_benchmark.h"


/**
 * @class BaselineBenchmark
 * @brief Baseline Benchmark implementation
 */
class BaselineBenchmark : public ECCBenchmark {
public:
  explicit BaselineBenchmark(const BenchmarkConfig& config) noexcept;
  ~BaselineBenchmark() noexcept = default;

  int setup() noexcept override;
  void teardown() noexcept override;
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
  bool check_for_corruption() const noexcept override;

private:
  // Data Buffers
  uint8_t* original_buffer_ = nullptr;    ///< Buffer for the original data we want to transmit
  uint8_t* parity_block_ = nullptr;       ///< Buffer for the decoded data
};


#endif