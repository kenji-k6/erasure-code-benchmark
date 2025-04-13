#ifndef CM256_BM_HPP
#define CM256_BM_HPP

#include "abstract_bm.hpp"
#include "cm256.h"

/**
 * @class CM256Benchmark
 * @brief Benchmark implementation for the CM256 EC library https://github.com/catid/cm256
 * 
 * This class implements the AbstractBenchmark interface, providing specific functionality
 * for benchmarking the CM256 library. It supports setup, teardown, encoding, decoding,
 * data loss simulation and corruption checking.
 */
class CM256Benchmark : public AbstractBenchmark {
public:
  explicit CM256Benchmark(const BenchmarkConfig& config) noexcept;
  ~CM256Benchmark() noexcept override;
  
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;

private:
  // Data Buffers
  uint8_t* m_parity_buf;         ///< Buffer for parity data

  // CM256 Internals
  cm256_encoder_params m_params;       ///< cm256 internal parameters
  std::vector<cm256_block> m_blocks;   ///< vector of cm256 blocks (keeps track of pointers and indices)
};

#endif // CM256_BM_HPP