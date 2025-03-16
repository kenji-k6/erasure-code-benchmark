#ifndef LEOPARD_BENCHMARK_HPP
#define LEOPARD_BENCHMARK_HPP

#include "abstract_benchmark.hpp"


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
  size_t m_encode_work_count = 0;
  size_t m_decode_work_count = 0;

  // Data Buffers
  std::unique_ptr<uint8_t[]> m_original_buffer;  ///< Buffer for the original data we want to transmit
  std::unique_ptr<uint8_t[]> m_encode_buffer;    ///< Buffer for the encoded data
  std::unique_ptr<uint8_t[]> m_decode_buffer;    ///< Buffer for the decoded data

  // Pointer vectors
  std::vector<uint8_t*> m_original_ptrs;       ///< Pointers to the original data blocks
  std::vector<uint8_t*> m_encode_work_ptrs;    ///< Pointers to the encoded data blocks
  std::vector<uint8_t*> m_decode_work_ptrs;    ///< Pointers to the decoded data blocks
};

#endif // LEOPARD_BENCHMARK_HPP