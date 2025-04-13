#ifndef XOREC_GPU_CMP_BM_HPP
#define XOREC_GPU_CMP_BM_HPP

#include "abstract_bm.hpp"

class XorecBenchmarkGpuCmp : public AbstractBenchmark {
public:
  explicit XorecBenchmarkGpuCmp(const BenchmarkConfig& config) noexcept;
  ~XorecBenchmarkGpuCmp() noexcept override;
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
  bool check_for_corruption() const noexcept override;

protected:
  uint8_t* m_parity_buf;              ///< Buffer for the decoded data (allocated on unified memory)
};

#endif // XOREC_GPU_CMP_BM_HPP