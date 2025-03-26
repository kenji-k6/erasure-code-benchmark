#ifndef ABSTRACT_BM_HPP
#define ABSTRACT_BM_HPP

#include "bm_config.hpp"
#include <immintrin.h>
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
  virtual bool check_for_corruption() const noexcept {
    for (unsigned i = 0; i < m_size_msg/m_size_blk; ++i) {
      uint8_t* data_ptr = m_data_buffer + i*m_size_data_submsg;
      if (!validate_block(data_ptr, m_size_blk)) return false;
    }
    return true;
  }

protected:
  explicit ECBenchmark(const BenchmarkConfig& config) noexcept
    : m_size_msg(config.message_size),
      m_size_blk(config.block_size),
      m_fec_params(config.fec_params),
      m_num_lst_rdma_pkts(config.num_lost_rmda_packets),
      m_lst_rdma_pkts(config.lost_rdma_packets)
  {
    m_size_data_submsg = get<0>(config.fec_params)*config.block_size;
    m_size_parity_submsg = get<1>(config.fec_params)*config.block_size;

    m_num_chunks = config.message_size/m_size_data_submsg;
    m_blks_per_chunk = get<0>(config.fec_params)+get<1>(config.fec_params);
  }




  size_t m_size_msg;
  size_t m_size_blk;
  FECTuple m_fec_params;

  size_t m_num_lst_rdma_pkts;
  const std::vector<uint32_t>& m_lst_rdma_pkts;

  
  size_t m_size_data_submsg;
  size_t m_size_parity_submsg;
  
  size_t m_blks_per_chunk;
  size_t m_num_chunks;

  uint8_t* m_data_buffer;
  uint8_t* m_block_bitmap;
};

#endif // ABSTRACT_BM_HPP