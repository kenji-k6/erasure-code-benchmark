#ifndef XOREC_GPU_PTR_BENCHMARK_HPP
#define XOREC_GPU_PTR_BENCHMARK_HPP

#include "abstract_benchmark.hpp"


/**
 * @class XorecBenchmarkGPU
 * @brief Xorec Benchmark implementation using unified memory, computation still happens on the CPU
 */
class XorecBenchmarkGPUPtr : public ECBenchmark {
public:
  explicit XorecBenchmarkGPUPtr(const BenchmarkConfig& config) noexcept;
  ~XorecBenchmarkGPUPtr() noexcept override;
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
  bool check_for_corruption() const noexcept override;
  void touch_gpu_memory() noexcept override;

protected:

  uint32_t m_num_total_blocks;
  // Data Buffers
  uint8_t *m_data_buffer;          ///< Buffer for the original data we want to transmit
  std::unique_ptr<uint8_t[]> m_parity_buffer;        ///< Buffer for the decoded data
  std::unique_ptr<uint8_t[]> m_block_bitmap; ///< Bitmap to check if all data arrived

  XorecVersion m_version;
  bool m_prefetch;
};

#endif // XOREC_GPU_PTR_BENCHMARK_HPP