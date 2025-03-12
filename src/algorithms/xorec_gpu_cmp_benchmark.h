#ifndef XOREC_GPU_CMP_BENCHMARK_H
#define XOREC_GPU_CMP_BENCHMARK_H

#include "abstract_benchmark.h"

class XorecBenchmarkGPUCmp : public ECBenchmark {
public:
  explicit XorecBenchmarkGPUCmp(const BenchmarkConfig& config) noexcept;
  ~XorecBenchmarkGPUCmp() noexcept = default;
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
  bool check_for_corruption() const noexcept override;
  void touch_gpu_memory() noexcept override;

protected:
  uint32_t num_total_blocks_;
  //  Buffers & Bitmap
  uint8_t *data_buffer_;                ///< Buffer for the original data we want to transmit (allocated on unified memory)
  uint8_t *parity_buffer_;              ///< Buffer for the decoded data (allocated on unified memory)
  std::unique_ptr<uint8_t[]> block_bitmap_; ///< Bitmap to check if all data arrived
};

#endif // XOREC_GPU_CMP_BENCHMARK_H