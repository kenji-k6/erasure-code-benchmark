#ifndef XOREC_BENCHMARK_H
#define XOREC_BENCHMARK_H

#include "abstract_benchmark.h"


/**
 * @class XorecBenchmark
 * @brief Xorec Benchmark implementation
 */
class XorecBenchmark : public ECBenchmark {
public:
  ~XorecBenchmark() noexcept override = default;
  void simulate_data_loss() noexcept override;
  bool check_for_corruption() const noexcept override;

protected:
  explicit XorecBenchmark(const BenchmarkConfig& config) noexcept;
  uint32_t num_total_blocks_;
  // Data Buffers
  std::unique_ptr<uint8_t[]> data_buffer_;  ///< Buffer for the original data we want to transmit
  std::unique_ptr<uint8_t[]> parity_buffer_;///< Buffer for the decoded data
  std::unique_ptr<uint8_t[]> block_bitmap_; ///< Bitmap to check if all data arrived
};


/**
 * @class XorecBenchmarkScalar
 * @brief Xorec Benchmark implementation using scalar operations
 */
class XorecBenchmarkScalar : public XorecBenchmark {
public:
  explicit XorecBenchmarkScalar(const BenchmarkConfig& config) noexcept;

  int encode() noexcept override;
  int decode() noexcept override;
};


/**
 * @class XorecBenchmarkAVX
 * @brief Xorec Benchmark implementation using AVX operations
 */
class XorecBenchmarkAVX : public XorecBenchmark {
  public:
    explicit XorecBenchmarkAVX(const BenchmarkConfig& config) noexcept;
  
    int encode() noexcept override;
    int decode() noexcept override;
};


/**
 * @class XorecBenchmarkAVX
 * @brief Xorec Benchmark implementation using AVX operations
 */
class XorecBenchmarkAVX2 : public XorecBenchmark {
public:
  explicit XorecBenchmarkAVX2(const BenchmarkConfig& config) noexcept;

  int encode() noexcept override;
  int decode() noexcept override;
};

#endif // XOREC_BENCHMARK_H