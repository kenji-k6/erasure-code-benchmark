#ifndef XOREC_GPU_CMP_BM_HPP
#define XOREC_GPU_CMP_BM_HPP

#include "abstract_bm.hpp"

class XorecBenchmarkGpuCmp : public AbstractBenchmark {
public:
  explicit XorecBenchmarkGpuCmp(const BenchmarkConfig& config) noexcept;
  void setup() noexcept override;
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
  bool check_for_corruption() const noexcept override;

protected:
  void m_write_data_buffer() noexcept override;

private:
  std::unique_ptr<uint8_t[], DeleterFunc<uint8_t>> m_gpu_block_bitmap; ///< Buffer for the original data we want to transmit
  size_t m_num_gpu_blocks;
  size_t m_threads_per_gpu_block;
};

#endif // XOREC_GPU_CMP_BM_HPP