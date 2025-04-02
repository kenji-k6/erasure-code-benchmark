/**
 * @file isal_bm.cpp
 * @brief Benchmark implementation for the Intel ISA-L library's EC implementation
 * 
 * Documentation can be found in isal_bm.h and abstract_bm.h
 */


#include "isal_bm.hpp"

#include "erasure_code.h"
#include "utils.hpp"


ISALBenchmark::ISALBenchmark(const BenchmarkConfig& config) noexcept : ECBenchmark(config) {

  // Allocate matrices etc.
  m_encode_matrix_vec.reserve(m_num_chunks);
  m_decode_matrix_vec.reserve(m_num_chunks);
  m_invert_matrix_vec.reserve(m_num_chunks);
  m_temp_matrix_vec.reserve(m_num_chunks);
  m_g_tbls_vec.reserve(m_num_chunks);
  m_frag_ptrs_vec.reserve(m_num_chunks);
  m_parity_src_ptrs_vec.reserve(m_num_chunks);
  m_recovery_outp_ptrs_vec.reserve(m_num_chunks);
  m_block_err_list_vec.reserve(m_num_chunks);
  m_decode_index_vec.reserve(m_num_chunks);

  for (unsigned i = 0; i < m_num_chunks; ++i) {
    m_encode_matrix_vec[i] = reinterpret_cast<uint8_t*>(_mm_malloc(m_blks_per_chunk * m_data_blks_per_chunk, ALIGNMENT)); 
    m_decode_matrix_vec[i] = reinterpret_cast<uint8_t*>(_mm_malloc(m_blks_per_chunk * m_data_blks_per_chunk, ALIGNMENT));
    m_invert_matrix_vec[i] = reinterpret_cast<uint8_t*>(_mm_malloc(m_blks_per_chunk * m_data_blks_per_chunk, ALIGNMENT));
    m_temp_matrix_vec[i] = reinterpret_cast<uint8_t*>(_mm_malloc(m_blks_per_chunk * m_data_blks_per_chunk, ALIGNMENT));
    m_g_tbls_vec[i] = reinterpret_cast<uint8_t*>(_mm_malloc(m_blks_per_chunk * m_data_blks_per_chunk * 32, ALIGNMENT));

    if (!m_encode_matrix_vec[i] || !m_decode_matrix_vec[i] || !m_invert_matrix_vec[i] || !m_temp_matrix_vec[i] || !m_g_tbls_vec[i]) {
      throw_error("ISAL: Failed to allocate memory.");
    }
  }

  m_data_buffer = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_chunks * m_size_data_submsg, ALIGNMENT));
  m_parity_buffer = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_chunks * m_size_parity_submsg, ALIGNMENT));
  m_recovery_outp_buffer = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_chunks * m_size_parity_submsg, ALIGNMENT));
  m_block_bitmap = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_chunks*m_blks_per_chunk, ALIGNMENT));

  if (!m_data_buffer || !m_recovery_outp_buffer || !m_block_bitmap || !m_parity_buffer) {
      throw_error("ISAL: Failed to allocate memory.");
    }

  memset(m_block_bitmap, 1, m_num_chunks * m_blks_per_chunk);
  for (unsigned i = 0; i < m_num_chunks; ++i) {
    uint8_t* local_data_buf = m_data_buffer + i * m_size_data_submsg;
    uint8_t* local_parity_buf = m_parity_buffer + i * m_size_parity_submsg;
    uint8_t* local_recovery_outp_buf = m_recovery_outp_buffer + i * m_size_parity_submsg;

    for (unsigned j = 0; j < m_data_blks_per_chunk; ++j) {
      m_frag_ptrs_vec[i][j] =  local_data_buf + j * m_size_blk;
    }
    for (unsigned j = 0; j < m_parity_blks_per_chunk; ++j) {
      m_frag_ptrs_vec[i][m_data_blks_per_chunk + j] = local_parity_buf + j * m_size_blk;
      m_recovery_outp_ptrs_vec[i][j] = local_recovery_outp_buf + j * m_size_blk;
    }
  }
  // Generate encode matricx, can be precomputed as it is fixed for a given m_num_original_blocks
  for (unsigned i = 0; i < m_num_chunks; ++i) {

    gf_gen_cauchy1_matrix(m_encode_matrix_vec[i], m_blks_per_chunk, m_data_blks_per_chunk);
    ec_init_tables(m_data_blks_per_chunk, m_parity_blks_per_chunk, &m_encode_matrix_vec[i][m_data_blks_per_chunk * m_data_blks_per_chunk], m_g_tbls_vec[i]);
  }
  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < m_size_msg/m_size_blk; ++i) {
    if (write_validation_pattern(i, m_data_buffer+i*m_size_blk, m_size_blk)) throw_error("ISAL: Failed to write validation pattern");
  }
}

ISALBenchmark::~ISALBenchmark() noexcept {
  for (unsigned i = 0; i < m_num_chunks; ++i) {
    _mm_free(m_encode_matrix_vec[i]);
    _mm_free(m_decode_matrix_vec[i]);
    _mm_free(m_invert_matrix_vec[i]);
    _mm_free(m_temp_matrix_vec[i]);
    _mm_free(m_g_tbls_vec[i]);
  }
  _mm_free(m_data_buffer);
  _mm_free(m_parity_buffer);
  _mm_free(m_recovery_outp_buffer);
  _mm_free(m_block_bitmap);
}

int ISALBenchmark::encode() noexcept {
  #pragma omp parallel for
  for (unsigned i = 0; i < m_num_chunks; ++i) {
    uint8_t* g_tbls = m_g_tbls_vec[i];
    auto frag_ptrs = m_frag_ptrs_vec[i];

    ec_encode_data(m_size_blk, m_data_blks_per_chunk, m_parity_blks_per_chunk, g_tbls, frag_ptrs.data(), frag_ptrs.data() + m_data_blks_per_chunk);
  }
  return 0;
}


int ISALBenchmark::decode() noexcept {
  for (unsigned i = 0; i < m_num_chunks; ++i) {
    uint8_t* block_bitmap = m_block_bitmap + i * m_blks_per_chunk;
    uint8_t* recovery_outp_buf = m_recovery_outp_buffer + i * m_size_parity_submsg;
    uint8_t* encode_matrix = m_encode_matrix_vec[i];
    uint8_t* decode_matrix = m_decode_matrix_vec[i];
    uint8_t* invert_matrix = m_invert_matrix_vec[i];
    uint8_t* temp_matrix = m_temp_matrix_vec[i];
    uint8_t* g_tbls = m_g_tbls_vec[i];
    auto frag_ptrs = m_frag_ptrs_vec[i];
    auto parity_src_ptrs = m_parity_src_ptrs_vec[i];
    auto recovery_outp_ptrs = m_recovery_outp_ptrs_vec[i];
    auto block_err_list = m_block_err_list_vec[i];
    auto decode_index = m_decode_index_vec[i];

    size_t nerrs = 0;
    for (unsigned j = 0; j < m_blks_per_chunk; ++j) {
      if (!block_bitmap[j]) block_err_list[nerrs++] = static_cast<uint8_t>(j);
    }

    if (nerrs == 0) continue;

    if (gf_gen_decode_matrix_simple(
      encode_matrix,
      decode_matrix,
      invert_matrix,
      temp_matrix,
      decode_index.data(),
      block_err_list.data(),
      nerrs, m_data_blks_per_chunk, m_parity_blks_per_chunk
    )) {
      return 1;
    }

    for (unsigned j = 0; j < m_data_blks_per_chunk; ++j) {
      parity_src_ptrs[j] = frag_ptrs[decode_index[j]];
    }

    ec_init_tables(m_data_blks_per_chunk, nerrs, decode_matrix, g_tbls);
    ec_encode_data(m_size_blk, m_data_blks_per_chunk, nerrs, g_tbls, parity_src_ptrs.data(), recovery_outp_ptrs.data());

    for (unsigned j = 0; j < nerrs; ++j) {
      if (block_err_list[j] < m_data_blks_per_chunk) {
        memcpy(frag_ptrs[block_err_list[j]], recovery_outp_buf + j * m_size_blk, m_size_blk);
      }
    }
  }
  return 0;
}


void ISALBenchmark::simulate_data_loss() noexcept {
  // TODO
  return;
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