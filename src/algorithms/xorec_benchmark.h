#ifndef XOREC_BENCHMARK_H
#define XOREC_BENCHMARK_H

#include "abstract_benchmark.h"
#include <bitset>


/**
 * @class XORECBenchmark
 * @brief XOREC Benchmark implementation
 */
class XORECBenchmark : public ECBenchmark {
public:
  explicit XORECBenchmark(const BenchmarkConfig& config) noexcept;
  ~XORECBenchmark() noexcept = default;
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
  bool check_for_corruption() const noexcept override;

protected:
  uint32_t num_total_blocks_;
  // Data Buffers
  std::unique_ptr<uint8_t[]> data_buffer_;  ///< Buffer for the original data we want to transmit
  std::unique_ptr<uint8_t[]> parity_buffer_;///< Buffer for the decoded data
  std::bitset<256> block_bitmap_;         ///< Bitmap to check if all data arrived
};


/**
 * @class XORECScalarBenchmark
 * @brief XOREC Benchmark implementation using scalar operations
 */
class XORECScalarBenchmark : public XORECBenchmark {
public:
  explicit XORECScalarBenchmark(const BenchmarkConfig& config) noexcept;

  int encode() noexcept override;
  int decode() noexcept override;
};


/**
 * @class XORECAVXBenchmark
 * @brief XOREC Benchmark implementation using AVX operations
 */
class XORECAVXBenchmark : public XORECBenchmark {
  public:
    explicit XORECAVXBenchmark(const BenchmarkConfig& config) noexcept;
  
    int encode() noexcept override;
    int decode() noexcept override;
};


/**
 * @class XORECAVXBenchmark
 * @brief XOREC Benchmark implementation using AVX operations
 */
class XORECAVX2Benchmark : public XORECBenchmark {
public:
  explicit XORECAVX2Benchmark(const BenchmarkConfig& config) noexcept;

  int encode() noexcept override;
  int decode() noexcept override;
};


/**
 * @class XORECBenchmarkGPU
 * @brief XOREC Benchmark implementation using unified memory and CUDA
 */
class XORECBenchmarkGPU : public ECBenchmark {
public:
  explicit XORECBenchmarkGPU(const BenchmarkConfig& config) noexcept;
  ~XORECBenchmarkGPU() noexcept override;
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
  bool check_for_corruption() const noexcept override;
  void make_memory_cold() noexcept override;

protected:
  uint32_t num_total_blocks_;
  // Data Buffers
  uint8_t *data_buffer_;          ///< Buffer for the original data we want to transmit
  std::unique_ptr<uint8_t[]> parity_buffer_;        ///< Buffer for the decoded data
  std::bitset<256> block_bitmap_;         ///< Bitmap to check if all data arrived
};


class XORECScalarBenchmarkGPU : public XORECBenchmarkGPU {
public:
  explicit XORECScalarBenchmarkGPU(const BenchmarkConfig& config) noexcept;

  int encode() noexcept override;
  int decode() noexcept override;
};


class XORECAVXBenchmarkGPU : public XORECBenchmarkGPU {
  public:
    explicit XORECAVXBenchmarkGPU(const BenchmarkConfig& config) noexcept;
  
    int encode() noexcept override;
    int decode() noexcept override;
};

class XORECAVX2BenchmarkGPU : public XORECBenchmarkGPU {
  public:
    explicit XORECAVX2BenchmarkGPU(const BenchmarkConfig& config) noexcept;
  
    int encode() noexcept override;
    int decode() noexcept override;
  };

#endif // XOREC_BENCHMARK_H