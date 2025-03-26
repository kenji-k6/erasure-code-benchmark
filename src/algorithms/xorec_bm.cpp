/**
 * @file xorec_bm.cpp
 * @brief Benchmark implementation for the XOR-EC implementation.
 * 
 * Documentation can be found in xorec.h and abstract_bm.h
 */


#include "xorec_bm.hpp"
#include "xorec.hpp"
#include "utils.hpp"


XorecBenchmark::XorecBenchmark(const BenchmarkConfig& config) noexcept : ECBenchmark(config) {
  xorec_init();
  m_data_buffer = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_chunks * m_size_data_submsg, ALIGNMENT));
  m_parity_buffer = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_chunks * m_size_parity_submsg, ALIGNMENT));
  m_block_bitmap = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_chunks * m_blks_per_chunk, ALIGNMENT));

  if (!m_data_buffer || !m_parity_buffer || !m_block_bitmap) throw_error("XorecBenchmark: Failed to allocate memory.");

  for (unsigned i = 0; i < m_size_msg/m_size_blk; ++i) {
    if (write_validation_pattern(i, m_data_buffer, m_size_blk)) throw_error("XorecBenchmark: Failed to write validation pattern");
  }
}

XorecBenchmark::~XorecBenchmark() noexcept {
  _mm_free(m_data_buffer);
  _mm_free(m_parity_buffer);
  _mm_free(m_block_bitmap);
}

int XorecBenchmark::encode() noexcept {
  for (unsigned i = 0; i < m_num_chunks; ++i) {
    uint8_t* data_ptr = m_data_buffer + i*m_size_data_submsg;
    uint8_t* parity_ptr = m_parity_buffer + i*m_size_parity_submsg;
    size_t fec_0 = get<0>(m_fec_params);
    size_t fec_1 = get<1>(m_fec_params);
    if (xorec_encode(data_ptr, parity_ptr, m_size_blk, fec_0, fec_1) != XorecResult::Success) return 1;
  }
  return 0;
}

int XorecBenchmark::decode() noexcept {
  for (unsigned i = 0; i < m_num_chunks; ++i) {
    uint8_t* data_ptr = m_data_buffer + i*m_size_data_submsg;
    uint8_t* parity_ptr = m_parity_buffer + i*m_size_parity_submsg;
    size_t fec_0 = get<0>(m_fec_params);
    size_t fec_1 = get<1>(m_fec_params);
    uint8_t* bitmap_ptr = m_block_bitmap + i*m_blks_per_chunk;

    if (xorec_decode(data_ptr, parity_ptr, m_size_blk, fec_0, fec_1, bitmap_ptr) != XorecResult::Success) return 1;
  }
}

void XorecBenchmark::simulate_data_loss() noexcept {
  return; // TODO
}

