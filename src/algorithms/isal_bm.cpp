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
  m_encode_matrix = reinterpret_cast<uint8_t*>(_mm_malloc(m_blks_per_chunk * m_data_blks_per_chunk, ALIGNMENT)); 
  m_decode_matrix = reinterpret_cast<uint8_t*>(_mm_malloc(m_blks_per_chunk * m_data_blks_per_chunk, ALIGNMENT));
  m_invert_matrix = reinterpret_cast<uint8_t*>(_mm_malloc(m_blks_per_chunk * m_data_blks_per_chunk, ALIGNMENT));
  m_temp_matrix = reinterpret_cast<uint8_t*>(_mm_malloc(m_blks_per_chunk * m_data_blks_per_chunk, ALIGNMENT));
  m_g_tbls = reinterpret_cast<uint8_t*>(_mm_malloc(m_blks_per_chunk * m_data_blks_per_chunk * 32, ALIGNMENT));

  m_data_buffer = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_chunks * m_size_data_submsg, ALIGNMENT));
  m_parity_buffer = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_chunks * m_size_parity_submsg, ALIGNMENT));

  m_recovery_outp_buffer = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_chunks * m_parity_blks_per_chunk * m_size_blk, ALIGNMENT));
  m_block_bitmap = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_chunks*m_blks_per_chunk, ALIGNMENT));
  if (!m_encode_matrix || !m_decode_matrix || !m_invert_matrix || !m_temp_matrix || !m_g_tbls ||
    !m_data_buffer || !m_recovery_outp_buffer || !m_block_bitmap || !m_parity_buffer) {
      throw_error("ISAL: Failed to allocate memory.");
    }
  memset(m_block_bitmap, 1, m_num_chunks * m_blks_per_chunk);

  // Initialize Pointer vectors
  m_frag_ptrs.reserve(ECLimits::ISAL_MAX_TOT_BLOCKS*m_num_chunks);

  m_parity_src_ptrs.reserve(ECLimits::ISAL_MAX_TOT_BLOCKS*m_num_chunks);
  m_recovery_outp_ptrs.reserve(ECLimits::ISAL_MAX_TOT_BLOCKS*m_num_chunks);
  m_decode_index.reserve(ECLimits::ISAL_MAX_TOT_BLOCKS*m_num_chunks);

  for (unsigned i = 0; i < m_num_chunks; ++i) {
    unsigned idx;
    for (idx = 0; idx < m_data_blks_per_chunk; ++idx) {
      m_frag_ptrs[i*ECLimits::ISAL_MAX_TOT_BLOCKS + idx] = m_data_buffer + i * m_size_data_submsg + idx * m_size_blk;
    }
    for (idx = 0; idx < m_parity_blks_per_chunk; ++idx) {
      m_frag_ptrs[i*ECLimits::ISAL_MAX_TOT_BLOCKS + m_data_blks_per_chunk + idx] = m_parity_buffer + i * m_size_parity_submsg + idx * m_size_blk;
    }
  }

  for (unsigned i = 0; i < m_num_chunks; ++i) {
    for (unsigned idx = 0; idx < m_parity_blks_per_chunk; ++idx) {
      m_recovery_outp_ptrs[i*ECLimits::ISAL_MAX_TOT_BLOCKS + idx] = m_data_buffer + i * m_size_parity_submsg + idx * m_size_blk; 
    }
  }

  // Generate encode matricx, can be precomputed as it is fixed for a given m_num_original_blocks
  gf_gen_cauchy1_matrix(m_encode_matrix, m_blks_per_chunk, m_data_blks_per_chunk);

  // Initialize generator tables for encoding, can be precomputed as it is fixed for a given m_num_original_blocks
  ec_init_tables(m_data_blks_per_chunk, m_parity_blks_per_chunk, &m_encode_matrix[m_data_blks_per_chunk * m_data_blks_per_chunk], m_g_tbls);

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < m_size_msg/m_size_blk; ++i) {
    if (write_validation_pattern(i, m_data_buffer+i*m_size_blk, m_size_blk)) throw_error("ISAL: Failed to write validation pattern");
  }
}

ISALBenchmark::~ISALBenchmark() noexcept {
  _mm_free(m_encode_matrix);
  _mm_free(m_decode_matrix);
  _mm_free(m_invert_matrix);
  _mm_free(m_temp_matrix);
  _mm_free(m_g_tbls);
  _mm_free(m_data_buffer);
  _mm_free(m_parity_buffer);
  _mm_free(m_recovery_outp_buffer);
  _mm_free(m_block_bitmap);
}

int ISALBenchmark::encode() noexcept {
  uint8_t** frag_ptrs = reinterpret_cast<uint8_t**>(m_frag_ptrs.data());

  for (unsigned i = 0; i < m_num_chunks; ++i) {
    ec_encode_data(m_size_blk, m_data_blks_per_chunk, m_parity_blks_per_chunk, m_g_tbls, frag_ptrs, frag_ptrs + m_data_blks_per_chunk);
    frag_ptrs += ECLimits::ISAL_MAX_TOT_BLOCKS;
  }
  return 0;
}


int ISALBenchmark::decode() noexcept {
  uint8_t** frag_ptrs = reinterpret_cast<uint8_t**>(m_frag_ptrs.data());
  uint8_t** parity_src_ptrs = m_parity_src_ptrs.data();
  uint8_t** recovery_outp_ptrs = m_recovery_outp_ptrs.data();
  uint8_t* decode_index = m_decode_index.data();
  uint8_t* bitmap_ptr = m_block_bitmap;

  for (unsigned i = 0; i < m_num_chunks; ++i) {

    m_block_err_list.clear();

    for (unsigned idx = 0; idx < m_blks_per_chunk; ++idx) {
      if (bitmap_ptr[idx] == 0) m_block_err_list.push_back(static_cast<uint8_t>(idx));
    }

    size_t num_lost_blocks = m_block_err_list.size();

    if (m_block_err_list.empty()) continue;

    if (gf_gen_decode_matrix_simple(m_encode_matrix, m_decode_matrix, m_invert_matrix,
                                    m_temp_matrix, decode_index, m_block_err_list, num_lost_blocks,
                                    m_data_blks_per_chunk, m_blks_per_chunk)) {
      return -1;
    }

    for (unsigned idx = 0; idx < m_data_blks_per_chunk; ++idx) {
      parity_src_ptrs[idx] = frag_ptrs[decode_index[idx]];
    }
    ec_init_tables(m_data_blks_per_chunk, num_lost_blocks, m_decode_matrix, m_g_tbls);
    ec_encode_data(m_size_blk, m_data_blks_per_chunk, num_lost_blocks, m_g_tbls, parity_src_ptrs, recovery_outp_ptrs);
    
    // for (unsigned i = 0; i < m_block_err_list.size(); ++i) {
    //   memcpy(orig_ptrs[m_block_err_list[i]], recovery_outp_ptrs[i], m_size_blk);
    // }
    frag_ptrs += ECLimits::ISAL_MAX_TOT_BLOCKS;
    parity_src_ptrs += ECLimits::ISAL_MAX_TOT_BLOCKS;
    recovery_outp_ptrs += ECLimits::ISAL_MAX_TOT_BLOCKS;
    decode_index += ECLimits::ISAL_MAX_TOT_BLOCKS;
    bitmap_ptr += m_blks_per_chunk;
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