/**
 * @file isal_benchmark.cpp
 * @brief Benchmark implementation for the Intel ISA-L library's EC implementation
 * 
 * Documentation can be found in isal_benchmark.h and abstract_benchmark.h
 */


#include "isal_benchmark.hpp"

#include "erasure_code.h"
#include "utils.hpp"


ISALBenchmark::ISALBenchmark(const BenchmarkConfig& config) noexcept : ECBenchmark(config) {
  m_num_total_blocks = m_num_original_blocks + m_num_recovery_blocks;
  // Allocate matrices etc.
  m_encode_matrix = std::make_unique<uint8_t[]>(m_num_total_blocks * m_num_original_blocks);
  m_decode_matrix = std::make_unique<uint8_t[]>(m_num_total_blocks * m_num_original_blocks);
  m_invert_matrix = std::make_unique<uint8_t[]>(m_num_total_blocks * m_num_original_blocks);
  m_temp_matrix = std::make_unique<uint8_t[]>(m_num_total_blocks * m_num_original_blocks);
  m_g_tbls = std::make_unique<uint8_t[]>(m_num_total_blocks * m_num_original_blocks * 32);

  m_original_buffer = std::make_unique<uint8_t[]>(m_block_size * m_num_total_blocks);
  m_recovery_outp_buffer = std::make_unique<uint8_t[]>(m_block_size * m_num_recovery_blocks);

  if (!m_encode_matrix || !m_decode_matrix || !m_invert_matrix || !m_temp_matrix || !m_g_tbls ||
      !m_original_buffer || !m_recovery_outp_buffer) {
    throw_error("ISAL: Failed to allocate memory.");
  }

  // Initialize Pointer vectors
  m_original_ptrs.resize(ECLimits::ISAL_MAX_TOT_BLOCKS);
  m_recovery_src_ptrs.resize(ECLimits::ISAL_MAX_TOT_BLOCKS);
  m_recovery_outp_ptrs.resize(ECLimits::ISAL_MAX_TOT_BLOCKS);
  m_decode_index.resize(ECLimits::ISAL_MAX_TOT_BLOCKS);

  for (unsigned i = 0; i < m_num_total_blocks; ++i) {
    m_original_ptrs[i] = &m_original_buffer[i * m_block_size];
  }

  for (unsigned i = 0; i < m_num_original_blocks; ++i) {
    m_recovery_outp_ptrs[i] = &m_recovery_outp_buffer[i * m_block_size];
  }

  // Generate encode matricx, can be precomputed as it is fixed for a given m_num_original_blocks
  gf_gen_cauchy1_matrix(m_encode_matrix.get(), m_num_total_blocks, m_num_original_blocks);

  // Initialize generator tables for encoding, can be precomputed as it is fixed for a given m_num_original_blocks
  ec_init_tables(m_num_original_blocks, m_num_recovery_blocks, &m_encode_matrix[m_num_original_blocks * m_num_original_blocks], m_g_tbls.get());

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < m_num_original_blocks; ++i) {
    int write_res = write_validation_pattern(i, m_original_ptrs[i], m_block_size);
    if (write_res) throw_error("ISAL: Failed to write random checking packet.");
  }
}

int ISALBenchmark::encode() noexcept {
  ec_encode_data(m_block_size, m_num_original_blocks, m_num_recovery_blocks,
                 m_g_tbls.get(), m_original_ptrs.data(), m_original_ptrs.data() + m_num_original_blocks);
  return 0;
}



int ISALBenchmark::decode() noexcept {
  if (m_num_lost_blocks == 0) return 0;

  // Copy lost block indices (in reality this would be done by iterating and checking for lost blocks)
  for (auto idx : m_lost_block_idxs) {
    m_block_err_list.push_back(static_cast<uint8_t>(idx));
  }
  // Generate decoding matrix
  if (gf_gen_decode_matrix_simple(m_encode_matrix, m_decode_matrix, m_invert_matrix,
                                  m_temp_matrix, m_decode_index, m_block_err_list,
                                  m_num_lost_blocks, m_num_original_blocks, m_num_total_blocks)) {
    return -1;
  }
  // Set up recovery pointers
  for (unsigned i = 0; i < m_num_original_blocks; ++i) {
    m_recovery_src_ptrs[i] = m_original_ptrs[m_decode_index[i]];
  }
  // Initialize tables and perform recovery
  ec_init_tables(m_num_original_blocks, m_num_lost_blocks, m_decode_matrix.get(), m_g_tbls.get());
  ec_encode_data(m_block_size, m_num_original_blocks, m_num_lost_blocks,
                 m_g_tbls.get(), m_recovery_src_ptrs.data(), m_recovery_outp_ptrs.data());
                 
  return 0;
}


void ISALBenchmark::simulate_data_loss() noexcept {
  for (auto idx : m_lost_block_idxs) {
    memset(m_original_ptrs[idx], 0, m_block_size);
    m_original_ptrs[idx] = nullptr;
    if (idx > m_num_original_blocks) {
      memset(m_recovery_outp_ptrs[idx - m_num_original_blocks], 0, m_block_size);
    }
  }
}


bool ISALBenchmark::check_for_corruption() const noexcept {
  unsigned loss_idx = 0;
  for (unsigned i = 0; i < m_num_original_blocks; i++) {
    bool res = false;
    if (loss_idx < m_num_lost_blocks && i == m_lost_block_idxs[loss_idx]) {
      res = validate_block(m_recovery_outp_ptrs[loss_idx], m_block_size);
      loss_idx++;
    } else {
      res = validate_block(m_original_ptrs[i], m_block_size);
    }

    if (!res) return false;
  }
  return true;
}


int gf_gen_decode_matrix_simple(
  const std::unique_ptr<uint8_t[]>& encode_matrix,
  std::unique_ptr<uint8_t[]>& decode_matrix,
  std::unique_ptr<uint8_t[]>& invert_matrix,
  std::unique_ptr<uint8_t[]>& temp_matrix,
  std::vector<uint8_t>& decode_index,
  std::vector<uint8_t>& frag_err_list,
  const int nerrs, const int k, [[maybe_unused]] const int m
) {
  int i, j, p, r;
  int nsrcerrs = 0;
  uint8_t s;
  uint8_t frag_in_err[ECLimits::ISAL_MAX_TOT_BLOCKS];
  memset(frag_in_err, 0, sizeof(frag_in_err));

  // Order the fragments in erasure for easier sorting
  for (i = 0; i < nerrs; i++) {
    if (frag_err_list[i] < k) {
      nsrcerrs++;
    }
    frag_in_err[frag_err_list[i]] = 1;
  }
  // Construct b (matrix that encoded remaining blocks) by removing erased rows
  for (i = 0, r = 0; i < k; i++, r++) {
    while (frag_in_err[r]) {
      r++;
    }
    for (j = 0; j < k; j++) {
      
      temp_matrix[k * i + j] = encode_matrix[k * r + j];
    }

    decode_index[i] = r;
  }
  // Invert matrix to get recovery matrix
  if (gf_invert_matrix(temp_matrix.get(), invert_matrix.get(), k) < 0) {
    return -1;
  }
  // Get decode matrix with only wanted recovery rows
  for (i = 0; i < nerrs; i++) {
    if (frag_err_list[i] < k) {
      for (j = 0; j < k; j++) {
        decode_matrix[k * i + j] = invert_matrix[k * frag_err_list[i] + j];
      }
    }
  }
  // For non-src (parity) erasures need to multiply encode matrix * invert
  for (p = 0; p < nerrs; p++) {
    if (frag_err_list[p] >= k) { // A parity err
      for (i = 0; i < k; i++) {
        s = 0;
        for (j = 0; j < k; j++) {
          s ^= gf_mul(invert_matrix[j * k + i], encode_matrix[k * frag_err_list[p] + j]);
        }
        decode_matrix[k * p + i] = s;
      }
    }
  }
  return 0;
}