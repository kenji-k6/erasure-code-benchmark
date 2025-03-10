#ifndef ABSTRACT_BENCHMARK_H
#define ABSTRACT_BENCHMARK_H

#include "benchmark_utils.h"
#include <cstddef>
#include <iostream>

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
  virtual void make_memory_cold() noexcept {};

protected:
  explicit ECBenchmark(const BenchmarkConfig& config) noexcept
    : block_size_(config.block_size),
      num_original_blocks_(config.computed.num_original_blocks),
      num_recovery_blocks_(config.computed.num_recovery_blocks),
      num_lost_blocks_(config.num_lost_blocks),
      lost_block_idxs_(config.lost_block_idxs) {};

  uint64_t block_size_;
  uint32_t num_original_blocks_;
  uint32_t num_recovery_blocks_;
  uint64_t num_lost_blocks_;
  const std::vector<uint32_t>& lost_block_idxs_;
};

#endif // ABSTRACT_BENCHMARK_H