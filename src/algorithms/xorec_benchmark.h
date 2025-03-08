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

  int setup() noexcept override;
  void teardown() noexcept override;
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
  bool check_for_corruption() const noexcept override;

protected:
  // Data Buffers
  uint8_t* data_buffer_ = nullptr;        ///< Buffer for the original data we want to transmit
  uint8_t* parity_buffer_ = nullptr;      ///< Buffer for the decoded data
  std::bitset<256> block_bitmap_;         ///< Bitmap to check if all data arrived
};


/**
 * @class XORECScalarBenchmark
 * @brief XOREC Benchmark implementation using scalar operations
 */
class XORECScalarBenchmark : public XORECBenchmark {
public:
  explicit XORECScalarBenchmark(const BenchmarkConfig& config) noexcept;
  ~XORECScalarBenchmark() noexcept = default; 

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
    ~XORECAVXBenchmark() noexcept = default; 
  
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
  ~XORECAVX2Benchmark() noexcept = default; 

  int encode() noexcept override;
  int decode() noexcept override;
};


/**
 * @class CUDA_XORECBenchmark
 * @brief XOREC Benchmark implementation using unified memory and CUDA
 */

class CUDA_XORECBenchmark : public XORECBenchmark {
public:
  explicit CUDA_XORECBenchmark(const BenchmarkConfig& config) noexcept;
  ~CUDA_XORECBenchmark() noexcept = default; 

  int setup() noexcept override;
  void teardown() noexcept override;
  void make_memory_cold() noexcept override;
};


class CUDA_XORECScalarBenchmark : public CUDA_XORECBenchmark {
public:
  explicit CUDA_XORECScalarBenchmark(const BenchmarkConfig& config) noexcept;
  ~CUDA_XORECScalarBenchmark() noexcept = default; 

  int encode() noexcept override;
  int decode() noexcept override;
};


class CUDA_XORECAVXBenchmark : public CUDA_XORECBenchmark {
  public:
    explicit CUDA_XORECAVXBenchmark(const BenchmarkConfig& config) noexcept;
    ~CUDA_XORECAVXBenchmark() noexcept = default; 
  
    int encode() noexcept override;
    int decode() noexcept override;
};

class CUDA_XORECAVX2Benchmark : public CUDA_XORECBenchmark {
  public:
    explicit CUDA_XORECAVX2Benchmark(const BenchmarkConfig& config) noexcept;
    ~CUDA_XORECAVX2Benchmark() noexcept = default; 
  
    int encode() noexcept override;
    int decode() noexcept override;
  };

#endif // XOREC_BENCHMARK_H