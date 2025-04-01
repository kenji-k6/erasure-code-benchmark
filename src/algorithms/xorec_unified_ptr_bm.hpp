#ifndef XOREC_UNIFIED_PTR_BM_HPP
#define XOREC_UNIFIED_PTR_BM_HPP

#include "abstract_bm.hpp"


/**
 * @class XorecBenchmarkUnifiedPtr
 * @brief Xorec Benchmark implementation using unified memory, computation still happens on the CPU
 */
class XorecBenchmarkUnifiedPtr : public ECBenchmark {
public:
  explicit XorecBenchmarkUnifiedPtr(const BenchmarkConfig& config) noexcept;
  ~XorecBenchmarkUnifiedPtr() noexcept override;
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
  void touch_unified_memory() noexcept override;

protected:
  // Data Buffers
  uint8_t *m_parity_buf;        ///< Buffer for the decoded data
  XorecVersion m_version;
};

#endif // XOREC_UNIFIED_PTR_BM_HPP