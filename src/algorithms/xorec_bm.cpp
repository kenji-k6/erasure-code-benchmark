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
    uint8_t* data_ptr = m_data_buffer + i * m_size_data_submsg;
    uint8_t* parity_ptr = m_parity_buffer + i * m_size_parity_submsg;

    if (xorec_encode(data_ptr, parity_ptr, m_size_blk, m_data_blks_per_chunk, m_parity_blks_per_chunk) != XorecResult::Success) {
      #pragma omp atomic write
      return_code = 1;
    };
  }
  return return_code;
}

int XorecBenchmark::decode() noexcept {
  uint8_t* data_ptr = m_data_buffer;
  uint8_t* parity_ptr = m_parity_buffer;
  uint8_t* bitmap_ptr = m_block_bitmap;

  for (unsigned i = 0; i < m_num_chunks; ++i) {
    if (xorec_decode(data_ptr, parity_ptr, m_size_blk, m_data_blks_per_chunk, m_parity_blks_per_chunk, bitmap_ptr) != XorecResult::Success) return 1;
    data_ptr += m_size_data_submsg;
    parity_ptr += m_size_parity_submsg;
    bitmap_ptr += m_blks_per_chunk;
  }
  return 0;
}

void XorecBenchmark::simulate_data_loss() noexcept {
  return; // TODO
}

