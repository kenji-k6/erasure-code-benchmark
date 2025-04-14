/**
 * @file isal_bm.cpp
 * @brief Benchmark implementation for the Intel ISA-L library's EC implementation
 * 
 * Documentation can be found in isal_bm.h and abstract_bm.h
 */


#include "isal_bm.hpp"

#include "erasure_code.h"
#include "utils.hpp"


ISALBenchmark::ISALBenchmark(const BenchmarkConfig& config) noexcept
  : AbstractBenchmark(config),
    m_recovery_outp_buf(make_unique_aligned<uint8_t>(m_num_data_blocks * m_block_size)),
    m_encode_matrix(make_unique_aligned<uint8_t>(m_num_tot_blocks * m_num_data_blocks)),
    m_decode_matrix(make_unique_aligned<uint8_t>(m_num_tot_blocks * m_num_data_blocks)),
    m_invert_matrix(make_unique_aligned<uint8_t>(m_num_tot_blocks * m_num_data_blocks)),
    m_temp_matrix(make_unique_aligned<uint8_t>(m_num_tot_blocks * m_num_data_blocks)),
    m_g_tbls(make_unique_aligned<uint8_t>(m_num_tot_blocks * m_num_data_blocks * 32))
{
  for (unsigned i = 0; i < m_num_data_blocks; ++i) m_frag_ptrs[i] = &m_data_buf[i*m_block_size];
  for (unsigned i = 0; i < m_num_parity_blocks; ++i) {
    m_frag_ptrs[m_num_data_blocks + i] = &m_parity_buf[i*m_block_size];
    m_recovery_outp_ptrs[i] = &m_recovery_outp_buf[i*m_block_size];
  }
  // Generate encode matricx, can be precomputed as it is fixed for a given m_num_original_blocks
  gf_gen_cauchy1_matrix(m_encode_matrix.get(), m_num_tot_blocks, m_num_data_blocks);
  
  // Initialize generator tables for encoding, can be precomputed as it is fixed for a given m_num_original_blocks
  ec_init_tables(m_num_data_blocks, m_num_parity_blocks, &m_encode_matrix[m_num_data_blocks*m_num_data_blocks], m_g_tbls.get());
  m_write_data_buffer();
}


int ISALBenchmark::encode() noexcept {
  ec_encode_data(
    m_block_size,
    m_num_data_blocks,
    m_num_parity_blocks,
    m_g_tbls.get(),
    m_frag_ptrs.data(),
    &m_frag_ptrs[m_num_data_blocks]
  );
  return 0;
}



int ISALBenchmark::decode() noexcept {
  size_t nerrs = 0;
  for (unsigned i = 0; i < m_num_tot_blocks; ++i) {
    if (!m_block_bitmap[i]) m_block_err_list[nerrs++] = static_cast<uint8_t>(i);
  }

  if (nerrs == 0) return 0;

  if (gf_gen_decode_matrix_simple(m_encode_matrix.get(), m_decode_matrix.get(), m_invert_matrix.get(),
                                  m_temp_matrix.get(), m_decode_index.data(), m_block_err_list.data(),
                                  nerrs, m_num_data_blocks, m_num_tot_blocks)) {
    return -1;
  }

  for (unsigned i = 0; i < m_num_data_blocks; ++i) {
    m_parity_src_ptrs[i] = m_frag_ptrs[m_decode_index[i]];
  }

  ec_init_tables(m_num_data_blocks, nerrs, m_decode_matrix.get(), m_g_tbls.get());
  ec_encode_data(m_block_size, m_num_data_blocks, nerrs,
                 m_g_tbls.get(), m_parity_src_ptrs.data(), m_recovery_outp_ptrs.data());
  
  for (unsigned i = 0; i < nerrs; ++i) {
    if (m_block_err_list[i] < m_num_data_blocks) {
      memcpy(m_frag_ptrs[m_block_err_list[i]], &m_recovery_outp_buf[i*m_block_size], m_block_size);
    }
  }
  return 0;
}

int gf_gen_decode_matrix_simple(
  const uint8_t* encode_matrix,
  uint8_t* decode_matrix,
  uint8_t* invert_matrix,
  uint8_t* temp_matrix,
  uint8_t* decode_index,
  uint8_t* frag_err_list,
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
  if (gf_invert_matrix(temp_matrix, invert_matrix, k) < 0) {
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