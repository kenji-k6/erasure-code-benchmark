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
    ~XorecBenchmarkGpuPtr() noexcept override;
    int encode() noexcept override;
    int decode() noexcept override;
    void simulate_data_loss() noexcept override;
    bool check_for_corruption() const noexcept override;
  
  protected:
    // Data Buffers
    uint8_t *m_gpu_data_buf;          ///< Buffer for the original data we want to transmit
    uint8_t *m_parity_buf;        ///< Buffer for the decoded data
    XorecVersion m_version;
  };

#endif // XOREC_GPU_PTR_BM_HPP