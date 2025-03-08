#ifndef CUDA_XOREC_BENCHMARK_H
#define CUDA_XOREC_BENCHMARK_H

#include "abstract_benchmark.h"
#include <bitset>


/**
 * @class XORECBenchmark
 * @brief XOREC Benchmark implementation
 */
class CudaXORECBenchmark : public ECBenchmark {
public:
  explicit CudaXORECBenchmark(const BenchmarkConfig& config) noexcept;
  ~CudaXORECBenchmark() noexcept = default;

  int setup() noexcept override;
  void teardown() noexcept override;
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
  bool check_for_corruption() const noexcept override;

private:
  // Data Buffers
  uint8_t* data_buffer_ = nullptr;        ///< Buffer for the original data we want to transmit
  uint8_t* parity_buffer_ = nullptr;      ///< Buffer for the decoded data
  std::bitset<256> block_bitmap_;         ///< Bitmap to check if all data arrived
};

#endif // CUDA_XOREC_BENCHMARK_H