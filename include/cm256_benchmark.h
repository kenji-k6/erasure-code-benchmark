#ifndef CM256_BENCHMARK_H
#define CM256_BENCHMARK_H

#include "abstract_benchmark.h"
#include "cm256.h"
#include <vector>


/**
 * @class CM256Benchmark
 * @brief Benchmark implementation for the CM256 ECC library https://github.com/catid/cm256
 * 
 * This class implements the ECCBenchmark interface, providing specific functionality
 * for benchmarking the CM256 library. It supports setup, teardown, encoding, decoding,
 * data loss simulation, corruption checking and cache flushing.
 */
class CM256Benchmark : public ECCBenchmark {
public:
CM256Benchmark() = default;
  ~CM256Benchmark() noexcept = default;

  int setup() noexcept override;
  void teardown() noexcept override;
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
  bool check_for_corruption() const noexcept override;
  void flush_cache() noexcept override;

private:
  size_t num_original_blocks_ = 0;
  size_t num_recovery_blocks_ = 0;
  size_t block_size_ = 0;

  // Data Buffers
  uint8_t* original_buffer_ = nullptr;    ///< Buffer for the original data we want to transmit
  uint8_t* decode_buffer_ = nullptr;      ///< Buffer for the decoded data

  // CM256 Internals
  cm256_encoder_params params_;       ///< cm256 internal parameters
  std::vector<cm256_block> blocks_;   ///< vector of cm256 blocks (keeps track of pointers and indices)
};

#endif // CM256_BENCHMARK_H