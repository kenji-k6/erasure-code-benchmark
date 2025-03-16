#ifndef ABSTRACT_BENCHMARK_HPP
#define ABSTRACT_BENCHMARK_HPP

#include "benchmark_config.hpp"
#include <cstring>

/**
 * @class ECBenchmark
 * @brief Abstract base class for Erasure Code (EC) benchmarking
 * 
 * This class defines the interface that all EC libraries should implement for benchmarking
 * their encoding, decoding and error-checking capabilities.
 * The class provides a clean and standardized way to set up, run and clean up EC benchmark tests.
 */
class ECBenchmark {
public:
  virtual ~ECBenchmark() noexcept = default; ///< Default virtual destructor
  /**
   * @brief Run the encoding process.
   * 
   * Derived classes should implement the encoding process for their specific EC algorithm
   * This should include the actual error-correction encoding process.
   * 
   * @return 0 on success, non-zero on failure.
   */
  virtual int encode() noexcept = 0;

  /**
   * @brief Run the decoding process (with simulated data loss).
   * 
   * Derived classes should implement the decoding process, considering the possibility of
   * data loss or corruption, as part of the benchmark testing.
   * 
   * @attention Unless absolutely necesseary, this function should not be responsible for
   * simulating the actual data loss.
   * 
   * @return 0 on success, non-zero on failure.
   */
  virtual int decode() noexcept = 0;

  /**
   * @brief Simulate data loss during transmission.
   * 
   * This function introduces data loss to the dataset to simulate erasure during
   * data transmission. The actual implementation of this function will vary
   * a lot based on the EC algorithm/library being tested.
   */
  virtual void simulate_data_loss() noexcept = 0;

  /**
   * @brief Check if there is any corruption in the decoded data.
   * 
   * This method should validate the integrity of the decoded data post-decoding
   * Refer to the validating block implementation in utils.h for an approach to do this.
   * 
   * @return True if the data is not corrupted, false otherwise.
   */
  virtual bool check_for_corruption() const noexcept = 0;

  /**
   * @brief Function to ensure the CPU memory is cold (only relevant for GPU memory benchmarks)
   * 
   * If not overwritten it doesn nothing.
   */
  virtual void touch_gpu_memory() noexcept {};

protected:
  explicit ECBenchmark(const BenchmarkConfig& config) noexcept
    : m_block_size(config.block_size),
      m_num_original_blocks(config.computed.num_original_blocks),
      m_num_recovery_blocks(config.computed.num_recovery_blocks),
      m_num_lost_blocks(config.num_lost_blocks),
      m_lost_block_idxs(config.lost_block_idxs) {};

  size_t m_block_size;
  size_t m_num_original_blocks;
  size_t m_num_recovery_blocks;
  size_t m_num_lost_blocks;
  const std::vector<uint32_t>& m_lost_block_idxs;
};

#endif // ABSTRACT_BENCHMARK_HPP