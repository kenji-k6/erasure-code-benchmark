#ifndef ISAL_BENCHMARK_H
#define ISAL_BENCHMARK_H

#include "abstract_benchmark.h"
#include "utils.h"
#include <cstdint>


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
  ~ISALBenchmark() noexcept = default;

  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
  bool check_for_corruption() const noexcept override;
  
private:
  uint32_t num_total_blocks_;

  // Data Buffers
  std::unique_ptr<uint8_t[]> original_buffer_; ///< Buffer for the original data we want to transmit
  std::unique_ptr<uint8_t[]> recovery_outp_buffer_; ///< Buffer for recovery of corrupted data

  // Data Block Pointers
  std::vector<uint8_t*> original_ptrs_;
  std::vector<uint8_t*> recovery_src_ptrs_;
  std::vector<uint8_t*> recovery_outp_ptrs_;
  

  // Erasure and Coefficient Matrices
  std::unique_ptr<uint8_t[]> encode_matrix_;
  std::unique_ptr<uint8_t[]> decode_matrix_;
  std::unique_ptr<uint8_t[]> invert_matrix_;
  std::unique_ptr<uint8_t[]> temp_matrix_;
  std::unique_ptr<uint8_t[]> g_tbls_; ///< Generator tables for encoding

  std::vector<uint8_t> block_err_list_; ///< Array containing the indices of lost blocks
  std::vector<uint8_t> decode_index_;  ///< Array containing the indices of the blocks to decode
};

// Helper function for generating the decode matrix (simple version, implementation from ISA-L Github repository)
int gf_gen_decode_matrix_simple(
  const std::unique_ptr<uint8_t[]>& encode_matrix,
  std::unique_ptr<uint8_t[]>& decode_matrix,
  std::unique_ptr<uint8_t[]>& invert_matrix,
  std::unique_ptr<uint8_t[]>& temp_matrix,
  std::vector<uint8_t>& decode_index,
  std::vector<uint8_t>& frag_err_list,
  const int nerrs, const int k, [[maybe_unused]] const int m
);

#endif // ISAL_BENCHMARK_H