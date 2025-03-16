#ifndef XOREC_BENCHMARK_HPP
#define XOREC_BENCHMARK_HPP

#include "abstract_benchmark.hpp"


/**
 * @class XorecBenchmark
 * @brief Benchmark implementation for the baseline XOR-EC implementation.
 * 
 * This class implements the ECBenchmark interface, providing specific functionality
 * for benchmarking the XOR-EC implementation. It supports setup, teardown, encoding, decoding,
 * data loss simulation and corruption checking..
 */
class XorecBenchmark : public ECBenchmark {
public:
explicit XorecBenchmark(const BenchmarkConfig& config) noexcept;
  ~XorecBenchmark() noexcept override = default;
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
  bool check_for_corruption() const noexcept override;

protected:
  size_t m_num_total_blocks;

  // Data Buffers
  std::unique_ptr<uint8_t[]> m_data_buffer;  ///< Buffer for the original data we want to transmit
  std::unique_ptr<uint8_t[]> m_parity_buffer;///< Buffer for the decoded data
  std::unique_ptr<uint8_t[]> m_block_bitmap; ///< Bitmap to check if all data arrived
  XorecVersion m_version;
};

#endif // XOREC_BENCHMARK_HPP