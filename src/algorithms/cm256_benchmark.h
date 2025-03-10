#ifndef CM256_BENCHMARK_H
#define CM256_BENCHMARK_H

#include "abstract_benchmark.h"
#include "utils.h"
#include "cm256.h"
#include <vector>


/**
 * @class CM256Benchmark
 * @brief Benchmark implementation for the CM256 EC library https://github.com/catid/cm256
 * 
 * This class implements the ECBenchmark interface, providing specific functionality
 * for benchmarking the CM256 library. It supports setup, teardown, encoding, decoding,
 * data loss simulation and corruption checking.
 */
class CM256Benchmark : public ECBenchmark {
public:
  explicit CM256Benchmark(const BenchmarkConfig& config) noexcept;
  ~CM256Benchmark() noexcept = default;

  int setup() noexcept override;
  void teardown() noexcept override;
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
  bool check_for_corruption() const noexcept override;

private:
  // Data Buffers
  std::unique_ptr<uint8_t[]> original_buffer;
  std::unique_ptr<uint8_t[]> decode_buffer;

  // CM256 Internals
  cm256_encoder_params params_;       ///< cm256 internal parameters
  std::vector<cm256_block> blocks_;   ///< vector of cm256 blocks (keeps track of pointers and indices)
};

#endif // CM256_BENCHMARK_H