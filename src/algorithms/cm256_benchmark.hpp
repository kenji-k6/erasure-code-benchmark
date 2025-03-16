#ifndef CM256_BENCHMARK_HPP
#define CM256_BENCHMARK_HPP

#include "abstract_benchmark.hpp"
#include "cm256.h"

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
  
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
  bool check_for_corruption() const noexcept override;

private:
  // Data Buffers
  std::unique_ptr<uint8_t[]> m_original_buffer;
  std::unique_ptr<uint8_t[]> m_decode_buffer;

  // CM256 Internals
  cm256_encoder_params m_params;       ///< cm256 internal parameters
  std::vector<cm256_block> m_blocks;   ///< vector of cm256 blocks (keeps track of pointers and indices)
};

#endif // CM256_BENCHMARK_HPP