#ifndef XOREC_UNIFIED_PTR_BM_HPP
#define XOREC_UNIFIED_PTR_BM_HPP

#include "abstract_bm.hpp"


/**
 * @class XorecBenchmarkUnifiedPtr
 * @brief Xorec Benchmark implementation using unified memory, computation still happens on the CPU
 */
class XorecBenchmarkUnifiedPtr : public AbstractBenchmark {
public:
  explicit XorecBenchmarkUnifiedPtr(const BenchmarkConfig& config) noexcept;
  int encode() noexcept override;
  int decode() noexcept override;
  void touch_unified_memory() noexcept override;

protected:
  // Data Buffers
  XorecVersion m_version;
};

#endif // XOREC_UNIFIED_PTR_BM_HPP