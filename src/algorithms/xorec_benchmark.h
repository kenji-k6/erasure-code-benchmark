#ifndef Xorec_BENCHMARK_H
#define Xorec_BENCHMARK_H

#include "abstract_benchmark.h"
#include <bitset>


/**
 * @class XorecBenchmark
 * @brief Xorec Benchmark implementation
 */
class XorecBenchmark : public ECBenchmark {
public:
  explicit XorecBenchmark(const BenchmarkConfig& config) noexcept;
  ~XorecBenchmark() noexcept = default;
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
  bool check_for_corruption() const noexcept override;

protected:
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


/**
 * @class XorecBenchmarkGPU
 * @brief Xorec Benchmark implementation using unified memory, computation still happens on the CPU
 */
class XorecBenchmarkGPUPointer : public ECBenchmark {
public:
  explicit XorecBenchmarkGPUPointer(const BenchmarkConfig& config) noexcept;
  ~XorecBenchmarkGPUPointer() noexcept override;
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
  bool check_for_corruption() const noexcept override;
  void touch_gpu_memory() noexcept override;

protected:
  uint32_t num_total_blocks_;
  // Data Buffers
  uint8_t *data_buffer_;          ///< Buffer for the original data we want to transmit
  std::unique_ptr<uint8_t[]> parity_buffer_;        ///< Buffer for the decoded data
  std::unique_ptr<uint8_t[]> block_bitmap_; ///< Bitmap to check if all data arrived
};


class XorecBenchmarkScalarGPUPointer : public XorecBenchmarkGPUPointer {
public:
  explicit XorecBenchmarkScalarGPUPointer(const BenchmarkConfig& config) noexcept;

  int encode() noexcept override;
  int decode() noexcept override;
};


class XorecBenchmarkAVXGPUPointer : public XorecBenchmarkGPUPointer {
  public:
    explicit XorecBenchmarkAVXGPUPointer(const BenchmarkConfig& config) noexcept;
  
    int encode() noexcept override;
    int decode() noexcept override;
};

class XorecBenchmarkAVX2GPUPointer : public XorecBenchmarkGPUPointer {
  public:
    explicit XorecBenchmarkAVX2GPUPointer(const BenchmarkConfig& config) noexcept;
  
    int encode() noexcept override;
    int decode() noexcept override;
  };


class XorecBenchmarkGPUComputation : public ECBenchmark {
  public:
  explicit XorecBenchmarkGPUComputation(const BenchmarkConfig& config) noexcept;
  ~XorecBenchmarkGPUComputation() noexcept = default;
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
  bool check_for_corruption() const noexcept override;

protected:
  uint32_t num_total_blocks_;
  //  Buffers & Bitmap
  uint8_t *data_buffer_;                ///< Buffer for the original data we want to transmit (allocated on unified memory)
  uint8_t *parity_buffer_;              ///< Buffer for the decoded data (allocated on unified memory)
  std::unique_ptr<uint8_t[]> block_bitmap_; ///< Bitmap to check if all data arrived
};
#endif // Xorec_BENCHMARK_H