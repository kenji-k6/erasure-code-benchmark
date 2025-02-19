#ifndef ISAL_BENCHMARK_H
#define ISAL_BENCHMARK_H

#include "abstract_benchmark.h"
#include "utils.h"
#include <cstdint>


/**
 * @class ISALBenchmark
 * @brief Benchmark implementation for the Intel ISA-L library's ECC implementation https://github.com/intel/isa-l
 * 
 * This class implements the ECCBenchmark interface, providing specific functionality
 * for benchmarking the ISA-L library. It supports setup, teardown, encoding, decoding,
 * data loss simulation, corruption checking and cache flushing.
 */
class ISALBenchmark : public ECCBenchmark {
public:
  ISALBenchmark() = default;
  ~ISALBenchmark() noexcept = default;

  int setup() noexcept override;
  void teardown() noexcept override;
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
  bool check_for_corruption() const noexcept override;
  void flush_cache() noexcept override;
  
private:
  size_t num_original_blocks_ = 0;
  size_t num_recovery_blocks_ = 0;
  size_t num_total_blocks_ = 0;
  size_t block_size_ = 0;
  size_t num_lost_blocks_ = 0;

  // Data Buffers
  uint8_t *original_buffer_ = nullptr; ///< Buffer for the original data we want to transmit
  uint8_t *recovery_outp_buffer_ = nullptr; ///< Buffer for recovery of corrupted data

  // Pointer Arrays
  uint8_t *original_ptrs_[ECCLimits::ISAL_MAX_TOT_BLOCKS] = { nullptr }; ///< Pointers to the original data blocks
  uint8_t *recovery_src_ptrs_[ECCLimits::ISAL_MAX_DATA_BLOCKS] = { nullptr }; ///< Pointers to the recovery source data blocks
  uint8_t *recovery_outp_ptrs_[ECCLimits::ISAL_MAX_DATA_BLOCKS] = { nullptr }; ///< Pointers to the recovery output data blocks

  // Erasure and Coefficient Matrices
  uint8_t *encode_matrix_ = nullptr;
  uint8_t *decode_matrix_ = nullptr;
  uint8_t *invert_matrix_ = nullptr;
  uint8_t *temp_matrix_ = nullptr;
  uint8_t *g_tbls_ = nullptr; ///< Generator tables for encoding

  uint8_t block_err_list_[ECCLimits::ISAL_MAX_TOT_BLOCKS] = { 0 }; ///< Array containing the indices of lost blocks
  uint8_t decode_index_[ECCLimits::ISAL_MAX_TOT_BLOCKS] = { 0 }; ///< Array containing the indices of the blocks to decode
};

// Helper function for generating the decode matrix (simple version, implementation from ISA-L Github repository)
static int gf_gen_decode_matrix_simple(uint8_t *encode_matrix, uint8_t *decode_matrix,
                                       uint8_t *invert_matrix, uint8_t *temp_matrix,
                                       uint8_t *decode_index, uint8_t *frag_err_list,
                                       int nerrs, int k, int m);

#endif // ISAL_BENCHMARK_H