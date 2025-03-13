#ifndef XOREC_BENCHMARK_HPP
#define XOREC_BENCHMARK_HPP

#include "abstract_benchmark.hpp"


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
  uint32_t m_num_total_blocks;
  // Data Buffers
  std::unique_ptr<uint8_t[]> m_data_buffer;  ///< Buffer for the original data we want to transmit
  std::unique_ptr<uint8_t[]> m_parity_buffer;///< Buffer for the decoded data
  std::unique_ptr<uint8_t[]> m_block_bitmap; ///< Bitmap to check if all data arrived
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

#endif // XOREC_BENCHMARK_HPP