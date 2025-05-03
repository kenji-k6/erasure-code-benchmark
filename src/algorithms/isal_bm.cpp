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
    m_recovery_outp_buf(make_unique_aligned<uint8_t>(m_chunks * m_chunk_data_size)),
    m_encode_matrix(make_unique_aligned<uint8_t>(m_chunk_tot_blocks * m_chunk_data_blocks))
{
  // size vectors appropriately
  m_decode_matrix_vec.reserve(m_chunks);
  m_invert_matrix_vec.reserve(m_chunks);
  m_temp_matrix_vec.reserve(m_chunks);
  m_g_tbls_vec.reserve(m_chunks);
  m_frag_ptrs_vec.reserve(m_chunks);
  m_parity_src_ptrs_vec.reserve(m_chunks);
  m_recovery_outp_ptrs_vec.reserve(m_chunks);
  m_block_err_list_vec.reserve(m_chunks);
  m_decode_index_vec.reserve(m_chunks);

  for (unsigned c = 0; c < m_chunks; ++c) {
    m_decode_matrix_vec[c] = make_unique_aligned<uint8_t>(m_chunk_tot_blocks * m_chunk_data_blocks);
    m_invert_matrix_vec[c] = make_unique_aligned<uint8_t>(m_chunk_tot_blocks * m_chunk_data_blocks);
    m_temp_matrix_vec[c] = make_unique_aligned<uint8_t>(m_chunk_tot_blocks * m_chunk_data_blocks);
    m_g_tbls_vec[c] = make_unique_aligned<uint8_t>(m_chunk_tot_blocks * m_chunk_data_blocks * 32);
  }
}

void ISALBenchmark::setup() noexcept {
  std::fill_n(m_block_bitmap.get(), m_chunks * m_chunk_tot_blocks, 1);
  gf_gen_cauchy1_matrix(m_encode_matrix.get(), m_chunk_tot_blocks, m_chunk_data_blocks);

  for (unsigned c = 0; c < m_chunks; ++c) {
    auto data_buf = m_data_buf.get() + c * m_chunk_data_size;
    auto parity_buf = m_parity_buf.get() + c * m_chunk_parity_size;
    auto recovery_outp_buf = m_recovery_outp_buf.get() + c * m_chunk_data_size;
    auto g_tbls = m_g_tbls_vec[c].get();
    auto frag_ptrs = m_frag_ptrs_vec[c];
    auto recovery_outp_ptrs = m_recovery_outp_ptrs_vec[c];

    for (unsigned i = 0; i < m_chunk_data_blocks; ++i) frag_ptrs[i] = &data_buf[i*m_block_size];
    for (unsigned i = 0; i < m_chunk_parity_blocks; ++i) {
      frag_ptrs[m_chunk_data_blocks + i] = &parity_buf[i*m_block_size];
      recovery_outp_ptrs[i] = &recovery_outp_buf[i*m_block_size];
    }
    ec_init_tables(m_chunk_data_blocks, m_chunk_parity_blocks, &m_encode_matrix[m_chunk_data_blocks*m_chunk_data_blocks], g_tbls);
  }
  m_write_data_buffer();
  omp_set_num_threads(m_threads);
}


int ISALBenchmark::encode() noexcept {
  #pragma omp parallel for
  for (unsigned c = 0; c < m_chunks; ++c) {
    auto g_tbls = m_g_tbls_vec[c].get();
    auto frag_ptrs = m_frag_ptrs_vec[c];
    ec_encode_data(
      m_block_size,
      m_chunk_data_blocks,
      m_chunk_parity_blocks,
      g_tbls,
      frag_ptrs.data(),
      &frag_ptrs[m_chunk_data_blocks]
    );
  }
  return 0;
}



int ISALBenchmark::decode() noexcept {
  int return_code = 0;

  #pragma omp parallel for
  for (unsigned c = 0; c < m_chunks; ++c) {
    auto bitmap = m_block_bitmap.get() + c * m_chunk_tot_blocks;
    auto recovery_outp_buf = m_recovery_outp_buf.get() + c * m_chunk_data_size;
    auto decode_matrix = m_decode_matrix_vec[c].get();
    auto invert_matrix = m_invert_matrix_vec[c].get();
    auto temp_matrix = m_temp_matrix_vec[c].get();
    auto g_tbls = m_g_tbls_vec[c].get();
    auto frag_ptrs = m_frag_ptrs_vec[c];
    auto parity_src_ptrs = m_parity_src_ptrs_vec[c];
    auto recovery_outp_ptrs = m_recovery_outp_ptrs_vec[c];
    auto block_err_list = m_block_err_list_vec[c];
    auto decode_index = m_decode_index_vec[c];

    size_t nerrs = 0;

    for (unsigned i = 0; i < m_chunk_tot_blocks; ++i) {
      if (!bitmap[i]) block_err_list[nerrs++] = static_cast<uint8_t>(i);
    }

    if (nerrs == 0) continue;

    if (gf_gen_decode_matrix_simple(
      m_encode_matrix.get(),
      decode_matrix,
      invert_matrix,
      temp_matrix,
      decode_index.data(),
      block_err_list.data(),
      nerrs, m_chunk_data_blocks, m_chunk_tot_blocks
    )) {
      #pragma omp atomic write
      return_code = 1;
    }

    for (unsigned i = 0; i < m_chunk_data_blocks; ++i) {
      parity_src_ptrs[i] = frag_ptrs[decode_index[i]];
    }

    ec_init_tables(m_chunk_data_blocks, nerrs, decode_matrix, g_tbls);
    ec_encode_data(
      m_block_size,
      m_chunk_data_blocks,
      nerrs,
      g_tbls,
      parity_src_ptrs.data(),
      recovery_outp_ptrs.data()
    );

    for (unsigned i = 0; i < nerrs; ++i) {
      if (block_err_list[i] < m_chunk_data_blocks) {
        memcpy(frag_ptrs[block_err_list[i]], &recovery_outp_buf[i*m_block_size], m_block_size);
      }
    }
  }
  return return_code;
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