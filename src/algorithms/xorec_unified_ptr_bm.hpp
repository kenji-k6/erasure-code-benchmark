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
  void setup() noexcept override;
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
  
private:
  /**
   * @brief Function to ensure the CPU memory is cold (i.e. not cached)
   */
  void m_touch_unified_memory() noexcept;
  XorecVersion m_version;
};

#endif // XOREC_UNIFIED_PTR_BM_HPP