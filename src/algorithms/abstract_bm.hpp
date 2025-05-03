#ifndef ABSTRACT_BM_HPP
#define ABSTRACT_BM_HPP

#include "bm_config.hpp"
#include "utils.hpp"
#include <immintrin.h>
#include <cstring>
#include <omp.h>

/**
 * @class AbstractBenchmark
 * @brief Abstract base class for Erasure Code (EC) benchmarking
 * 
 * This class defines the interface that all EC libraries should implement for benchmarking
 * their encoding, decoding and error-checking capabilities.
 * The class provides a clean and standardized way to set up, run and clean up EC benchmark tests.
 */
class AbstractBenchmark {
public:
  virtual ~AbstractBenchmark() noexcept = default; ///< Default virtual destructor
  virtual void setup() noexcept = 0; ///< Pure virtual function for setup

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
  virtual void simulate_data_loss() noexcept;

  /**
   * @brief Check if there is any corruption in the decoded data.
   * 
   * This method should validate the integrity of the decoded data post-decoding
   * Refer to the validating block implementation in utils.h for an approach to do this.
   * 
   * @return True if the data is not corrupted, false otherwise.
   */
  virtual bool check_for_corruption() const noexcept;

protected:
  explicit AbstractBenchmark(const BenchmarkConfig& config) noexcept;
  virtual void m_write_data_buffer() noexcept; ///< writes validation pattern to data buffer

  size_t m_threads;
  size_t m_message_size;
  size_t m_block_size;
  
  
  size_t m_chunk_data_blocks;
  size_t m_chunk_parity_blocks;
  size_t m_chunk_tot_blocks;

  size_t m_chunks;

  size_t m_chunk_data_size;
  size_t m_chunk_parity_size;

  size_t m_chunk_lost_blocks;

  std::unique_ptr<uint8_t[], DeleterFunc<uint8_t>> m_data_buf;
  std::unique_ptr<uint8_t[], DeleterFunc<uint8_t>> m_parity_buf;
  std::unique_ptr<uint8_t[], DeleterFunc<uint8_t>> m_block_bitmap;
};

#endif // ABSTRACT_BM_HPP