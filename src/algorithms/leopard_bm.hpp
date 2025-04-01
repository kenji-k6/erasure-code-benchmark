#ifndef LEOPARD_BM_HPP
#define LEOPARD_BM_HPP

#include "abstract_bm.hpp"


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
  ~LeopardBenchmark() noexcept override;

  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;

private:
  size_t m_encode_work_count = 0;
  size_t m_decode_work_count = 0;

  uint8_t* m_encode_buf;
  uint8_t* m_decode_buf;

  // Pointer vectors
  std::vector<uint8_t*> m_original_ptrs;       ///< Pointers to the original data blocks
  std::vector<uint8_t*> m_encode_work_ptrs;    ///< Pointers to the encoded data blocks
  std::vector<uint8_t*> m_decode_work_ptrs;    ///< Pointers to the decoded data blocks
};

#endif // LEOPARD_BM_HPP