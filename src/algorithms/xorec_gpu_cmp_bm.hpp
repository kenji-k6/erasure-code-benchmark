#ifndef XOREC_GPU_CMP_BM_HPP
#define XOREC_GPU_CMP_BM_HPP

#include "abstract_bm.hpp"

class XorecBenchmarkGpuCmp : public ECBenchmark {
public:
  explicit XorecBenchmarkGpuCmp(const BenchmarkConfig& config) noexcept;
  ~XorecBenchmarkGpuCmp() noexcept override;
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
  bool check_for_corruption() const noexcept override;
  
protected:
  size_t m_num_total_blocks;
  //  Buffers & Bitmap
  uint8_t *m_parity_buffer;
};


class XorecBenchmarkGpuCmpCpuParity : public ECBenchmark {
  public:
    explicit XorecBenchmarkGpuCmpCpuParity(const BenchmarkConfig& config) noexcept;
    ~XorecBenchmarkGpuCmpCpuParity() noexcept override;
    int encode() noexcept override;
    int decode() noexcept override;
    void simulate_data_loss() noexcept override;
    bool check_for_corruption() const noexcept override;
    
  protected:
    size_t m_num_total_blocks;
    //  Buffers & Bitmap
    uint8_t *m_parity_buffer;
    uint8_t *m_gpu_parity_buffer;
  };

#endif // XOREC_GPU_CMP_BM_HPP