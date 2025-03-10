#ifndef ALL_GPU_BENCHMARK_H
#define ALL_GPU_BENCHMARK_H

#include "abstract_benchmark.h"
#include <bitset>


/**
 * @class XORECBenchmark
 * @brief XOREC Benchmark implementation
 */
class GPUBenchmark : public ECBenchmark {
  public:
    explicit GPUBenchmark(const BenchmarkConfig& config) noexcept;
    ~GPUBenchmark() noexcept override;
    int encode() noexcept override;
    int decode() noexcept override;
    void simulate_data_loss() noexcept override;
    bool check_for_corruption() const noexcept override;
  
  protected:
    uint32_t num_total_blocks_;
    // Data Buffers
    uint8_t *data_buffer_;  ///< Buffer for the original data we want to transmit
    uint8_t *parity_buffer_; ///< Buffer for the decoded data
    std::bitset<256> block_bitmap_;         ///< Bitmap to check if all data arrived
  };

#endif