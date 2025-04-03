/**
 * @file xorec_bm.cpp
 * @brief Benchmark implementation for the XOR-EC implementation.
 * 
 * Documentation can be found in xorec.h and abstract_bm.h
 */


#include "xorec_bm.hpp"
#include "xorec.hpp"
#include "utils.hpp"
#include <omp.h>


XorecBenchmark::XorecBenchmark(const BenchmarkConfig& config) noexcept : ECBenchmark(config) {
  xorec_init(m_data_blks_per_chunk, m_parity_blks_per_chunk);
  m_data_buffer = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_chunks * m_size_data_submsg, ALIGNMENT));
  m_parity_buffer = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_chunks * m_size_parity_submsg, ALIGNMENT));
  m_block_bitmap = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_chunks * m_blks_per_chunk, ALIGNMENT));
  if (!m_data_buffer || !m_parity_buffer || !m_block_bitmap) throw_error("XorecBenchmark: Failed to allocate memory.");

  memset(m_block_bitmap, 1, m_num_chunks * m_blks_per_chunk);
  for (unsigned i = 0; i < m_size_msg/m_size_blk; ++i) {
    if (write_validation_pattern(i, m_data_buffer+i*m_size_blk, m_size_blk)) throw_error("XorecBenchmark: Failed to write validation pattern");
  }
}

XorecBenchmark::~XorecBenchmark() noexcept {
  _mm_free(m_data_buffer);
  _mm_free(m_parity_buffer);
  _mm_free(m_block_bitmap);
}

int XorecBenchmark::encode() noexcept {
  int return_code = 0;

  #pragma omp parallel for
  for (unsigned i = 0; i < m_num_chunks; ++i) {
    uint8_t* data_buf = m_data_buffer + i * m_size_data_submsg;
    uint8_t* parity_buf = m_parity_buffer + i * m_size_parity_submsg;

    if (xorec_encode(data_buf, parity_buf, m_size_blk, m_data_blks_per_chunk, m_parity_blks_per_chunk) != XorecResult::Success) {
      #pragma omp atomic write
      return_code = 1;
    };
  }
  return return_code;
}

int XorecBenchmark::decode() noexcept {
  int return_code = 0;

  #pragma omp parallel for
  for (unsigned c = 0; c < m_num_chunks; ++c) {
    uint8_t* data_buf = m_data_buffer + c * m_size_data_submsg;
    uint8_t* parity_buf = m_parity_buffer + c * m_size_parity_submsg;
    uint8_t* bitmap = m_block_bitmap + c * m_blks_per_chunk;

    XorecResult res = xorec_decode(data_buf, parity_buf, m_size_blk, m_data_blks_per_chunk, m_parity_blks_per_chunk, bitmap);
    if (res != XorecResult::Success) {
      #pragma omp atomic write
      return_code = 1;
    }
  }
  return return_code;
}

void XorecBenchmark::simulate_data_loss() noexcept {
  for (unsigned c = 0; c < m_num_chunks; ++c) {
    uint8_t* data_buf = m_data_buffer + c * m_size_data_submsg;
    uint8_t* parity_buf = m_parity_buffer + c * m_size_parity_submsg;
    uint8_t* bitmap = m_block_bitmap + c * m_blks_per_chunk;

    select_lost_block_idxs(m_data_blks_per_chunk, m_parity_blks_per_chunk, m_num_lst_rdma_pkts, bitmap);
    
    unsigned i;
    for (i = 0; i < m_data_blks_per_chunk; ++i) {
      if (!bitmap[i]) memset(data_buf + i * m_size_blk, 0, m_size_blk);
    }

    for (; i < m_blks_per_chunk; ++i) {
      if (!bitmap[i]) memset(parity_buf + (i - m_data_blks_per_chunk) * m_size_blk, 0, m_size_blk);
    }
  }
}

