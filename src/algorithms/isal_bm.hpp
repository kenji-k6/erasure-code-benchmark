#ifndef ISAL_BM_HPP
#define ISAL_BM_HPP

#include "abstract_bm.hpp"


/**
 * @class ISALBenchmark
 * @brief Benchmark implementation for the Intel ISA-L library's EC implementation https://github.com/intel/isa-l
 * 
 * This class implements the ECBenchmark interface, providing specific functionality
 * for benchmarking the ISA-L library. It supports setup, teardown, encoding, decoding,
 * data loss simulation and corruption checking..
 */
class ISALBenchmark : public ECBenchmark {
public:
  explicit ISALBenchmark(const BenchmarkConfig& config) noexcept;
  ~ISALBenchmark() noexcept override;

  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
  
private:
  // Data Buffers
  uint8_t* m_parity_buffer;
  uint8_t* m_recovery_outp_buffer; ///< Buffer for recovery of corrupted data
  
  std::vector<uint8_t*> m_encode_matrix_vec; ///< Array of encode matrices
  std::vector<uint8_t*> m_decode_matrix_vec; ///< Array of decode matrices
  std::vector<uint8_t*> m_invert_matrix_vec; ///< Array of invert matrices
  std::vector<uint8_t*> m_temp_matrix_vec;   ///< Array of temporary matrices
  std::vector<uint8_t*> m_g_tbls_vec;     ///< Array of generator tables

  // Data Block Pointers

  std::vector<std::array<uint8_t*, ECLimits::ISAL_MAX_TOT_BLOCKS>> m_frag_ptrs_vec;
  std::vector<std::array<uint8_t*, ECLimits::ISAL_MAX_TOT_BLOCKS>> m_parity_src_ptrs_vec;
  std::vector<std::array<uint8_t*, ECLimits::ISAL_MAX_TOT_BLOCKS>> m_recovery_outp_ptrs_vec;



  std::vector<std::array<uint8_t, ECLimits::ISAL_MAX_TOT_BLOCKS>> m_block_err_list_vec; ///< Array containing the indices of lost blocks
  std::vector<std::array<uint8_t, ECLimits::ISAL_MAX_TOT_BLOCKS>> m_decode_index_vec;  ///< Array containing the indices of the blocks to decode
};

// Helper function for generating the decode matrix (simple version, implementation from ISA-L Github repository)
int gf_gen_decode_matrix_simple(
  const uint8_t* encode_matrix,
  uint8_t* decode_matrix,
  uint8_t* invert_matrix,
  uint8_t* temp_matrix,
  uint8_t* decode_index,
  uint8_t* frag_err_list,
  const int nerrs, const int k, [[maybe_unused]] const int m
);

#endif // ISAL_BM_HPP