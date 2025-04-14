#ifndef XOREC_GPU_PTR_BM_HPP
#define XOREC_GPU_PTR_BM_HPP
#include "abstract_bm.hpp"

/**
 * @class XorecBenchmarkGPUPtr
 * @brief Xorec Benchmark implementation using GPU memory, computation still happens on the CPU
 */

 class XorecBenchmarkGpuPtr : public AbstractBenchmark {
  public:
    explicit XorecBenchmarkGpuPtr(const BenchmarkConfig& config) noexcept;
    void setup() noexcept override;
    int encode() noexcept override;
    int decode() noexcept override;
    void simulate_data_loss() noexcept override;
    bool check_for_corruption() const noexcept override;
  
  private:
    // Data Buffers
    XorecVersion m_version;
    std::unique_ptr<uint8_t[], DeleterFunc<uint8_t>> m_gpu_data_buf; ///< Buffer for the original data we want to transmit
  };

#endif // XOREC_GPU_PTR_BM_HPP