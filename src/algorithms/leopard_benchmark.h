#ifndef LEOPARD_BENCHMARK_H
#define LEOPARD_BENCHMARK_H

#include "abstract_benchmark.h"
#include <cstdint>
#include <vector>


/**
 * @class LeopardBenchmark
 * @brief Benchmark implementation for the Leopard EC library https://github.com/catid/leopard
 * 
 * This class implements the ECBenchmark interface, providing specific functionality
 * for benchmarking the Leopard library. It supports setup, teardown, encoding, decoding,
 * data loss simulation and corruption checking..
 */
class LeopardBenchmark : public ECBenchmark {
public:
  explicit LeopardBenchmark(const BenchmarkConfig& config) noexcept;
  ~LeopardBenchmark() noexcept = default;

  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
  bool check_for_corruption() const noexcept override;

private:
  uint32_t encode_work_count_ = 0;
  uint32_t decode_work_count_ = 0;

  // Data Buffers
  std::unique_ptr<uint8_t[]> original_buffer_;  ///< Buffer for the original data we want to transmit
  std::unique_ptr<uint8_t[]> encode_buffer_;    ///< Buffer for the encoded data
  std::unique_ptr<uint8_t[]> decode_buffer_;    ///< Buffer for the decoded data

  // Pointer vectors
  std::vector<uint8_t*> original_ptrs_;       ///< Pointers to the original data blocks
  std::vector<uint8_t*> encode_work_ptrs_;    ///< Pointers to the encoded data blocks
  std::vector<uint8_t*> decode_work_ptrs_;    ///< Pointers to the decoded data blocks
};

#endif // LEOPARD_BENCHMARK_H