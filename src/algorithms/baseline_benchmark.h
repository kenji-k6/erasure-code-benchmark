#ifndef BASELINE_BENCHMARK_H
#define BASELINE_BENCHMARK_H

#include "abstract_benchmark.h"
#include <bitset>


/**
 * @class BaselineBenchmark
 * @brief Baseline Benchmark implementation
 */
class BaselineBenchmark : public ECBenchmark {
public:
  explicit BaselineBenchmark(const BenchmarkConfig& config) noexcept;
  ~BaselineBenchmark() noexcept = default;

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
 * @class BaselineScalarBenchmark
 * @brief Baseline Benchmark implementation using scalar operations
 */
class BaselineScalarBenchmark : public BaselineBenchmark {
public:
  explicit BaselineScalarBenchmark(const BenchmarkConfig& config) noexcept;
  ~BaselineScalarBenchmark() noexcept = default; 

  int encode() noexcept override;
  int decode() noexcept override;
};

/**
 * @class BaselineScalarNoOptBenchmark
 * @brief Baseline Benchmark implementation using scalar operations
 */
class BaselineScalarNoOptBenchmark : public BaselineBenchmark {
  public:
    explicit BaselineScalarNoOptBenchmark(const BenchmarkConfig& config) noexcept;
    ~BaselineScalarNoOptBenchmark() noexcept = default; 
  
    int encode() noexcept override;
    int decode() noexcept override;
  };


/**
 * @class BaselineAVXBenchmark
 * @brief Baseline Benchmark implementation using AVX operations
 */
class BaselineAVXBenchmark : public BaselineBenchmark {
  public:
    explicit BaselineAVXBenchmark(const BenchmarkConfig& config) noexcept;
    ~BaselineAVXBenchmark() noexcept = default; 
  
    int encode() noexcept override;
    int decode() noexcept override;
};


/**
 * @class BaselineAVXBenchmark
 * @brief Baseline Benchmark implementation using AVX operations
 */
class BaselineAVX2Benchmark : public BaselineBenchmark {
  public:
    explicit BaselineAVX2Benchmark(const BenchmarkConfig& config) noexcept;
    ~BaselineAVX2Benchmark() noexcept = default; 
  
    int encode() noexcept override;
    int decode() noexcept override;
};

#endif // BASELINE_BENCHMARK_H