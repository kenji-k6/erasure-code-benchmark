#ifndef LEOPARD_BM_HPP
#define LEOPARD_BM_HPP

#include "abstract_bm.hpp"


/**
 * @class LeopardBenchmark
 * @brief Benchmark implementation for the Leopard EC library https://github.com/catid/leopard
 * 
 * This class implements the AbstractBenchmark interface, providing specific functionality
 * for benchmarking the Leopard library. It supports setup, teardown, encoding, decoding,
 * data loss simulation and corruption checking..
 */
class LeopardBenchmark : public AbstractBenchmark {
public:
  explicit LeopardBenchmark(const BenchmarkConfig& config) noexcept;
  int encode() noexcept override;
  int decode() noexcept override;

private:
  size_t m_parity_work_count;
  size_t m_recovery_work_count;

  std::unique_ptr<uint8_t[], DeleterFunc<uint8_t>> m_recovery_buf;

  // Pointer vectors
  std::vector<uint8_t*> m_data_ptrs;      ///< Pointers to the original data blocks
  std::vector<uint8_t*> m_parity_ptrs;    ///< Pointers to the encoded data blocks
  std::vector<uint8_t*> m_recovery_ptrs;  ///< Pointers to the decoded data blocks
};

#endif // LEOPARD_BM_HPP