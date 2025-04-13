#ifndef XOREC_BM_HPP
#define XOREC_BM_HPP

#include "abstract_bm.hpp"


/**
 * @class XorecBenchmark
 * @brief Benchmark implementation for the baseline XOR-EC implementation.
 * 
 * This class implements the AbstractBenchmark interface, providing specific functionality
 * for benchmarking the XOR-EC implementation. It supports setup, teardown, encoding, decoding,
 * data loss simulation and corruption checking..
 */
class XorecBenchmark : public AbstractBenchmark {
public:
explicit XorecBenchmark(const BenchmarkConfig& config) noexcept;
  ~XorecBenchmark() noexcept override;
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;

protected:
  // Data Buffers
  uint8_t *m_parity_buf; ///< Buffer for the decoded data
  XorecVersion m_version;
};

#endif // XOREC_BM_HPP