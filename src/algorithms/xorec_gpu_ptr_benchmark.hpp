#ifndef XOREC_GPU_PTR_BENCHMARK_HPP
#define XOREC_GPU_PTR_BENCHMARK_HPP

#include "abstract_benchmark.hpp"


/**
 * @class XorecBenchmarkGPU
 * @brief Xorec Benchmark implementation using unified memory, computation still happens on the CPU
 */
class XorecBenchmarkGPUPtr : public ECBenchmark {
public:
  ~XorecBenchmarkGPUPtr() noexcept override;
  void simulate_data_loss() noexcept override;
  bool check_for_corruption() const noexcept override;
  void touch_gpu_memory() noexcept override;

protected:
  explicit XorecBenchmarkGPUPtr(const BenchmarkConfig& config) noexcept;

  uint32_t num_total_blocks_;
  // Data Buffers
  uint8_t *data_buffer_;          ///< Buffer for the original data we want to transmit
  std::unique_ptr<uint8_t[]> parity_buffer_;        ///< Buffer for the decoded data
  std::unique_ptr<uint8_t[]> block_bitmap_; ///< Bitmap to check if all data arrived
};


class XorecBenchmarkScalarGPUPtr : public XorecBenchmarkGPUPtr {
public:
  explicit XorecBenchmarkScalarGPUPtr(const BenchmarkConfig& config) noexcept;

  int encode() noexcept override;
  int decode() noexcept override;
};


class XorecBenchmarkAVXGPUPtr : public XorecBenchmarkGPUPtr {
  public:
    explicit XorecBenchmarkAVXGPUPtr(const BenchmarkConfig& config) noexcept;
  
    int encode() noexcept override;
    int decode() noexcept override;
};

class XorecBenchmarkAVX2GPUPtr : public XorecBenchmarkGPUPtr {
  public:
    explicit XorecBenchmarkAVX2GPUPtr(const BenchmarkConfig& config) noexcept;
  
    int encode() noexcept override;
    int decode() noexcept override;
};

#endif // XOREC_GPU_PTR_BENCHMARK_HPP