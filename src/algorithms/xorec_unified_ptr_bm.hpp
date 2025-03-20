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
  bool check_for_corruption() const noexcept override;
  void touch_unified_memory() noexcept override;

protected:
  size_t m_num_total_blocks;
  // Data Buffers
  uint8_t *m_data_buffer;          ///< Buffer for the original data we want to transmit
  std::unique_ptr<uint8_t[]> m_parity_buffer;        ///< Buffer for the decoded data
  std::unique_ptr<uint8_t[]> m_block_bitmap; ///< Bitmap to check if all data arrived

  XorecVersion m_version;
  bool m_prefetch;
};

#endif // XOREC_UNIFIED_PTR_BM_HPP