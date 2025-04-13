/**
 * @file xorec_bm.cpp
 * @brief Benchmark implementation for the XOR-EC implementation.
 * 
 * Documentation can be found in xorec.h and abstract_bm.h
 */


#include "xorec_bm.hpp"
#include "xorec.hpp"
#include "utils.hpp"
 
 
XorecBenchmark::XorecBenchmark(const BenchmarkConfig& config) noexcept : AbstractBenchmark(config) {
  xorec_init(m_num_data_blocks);
  m_version = config.xorec_params.version;

  m_data_buf = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_data_blocks*m_block_size, ALIGNMENT));
  m_parity_buf = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_parity_blocks*m_block_size, ALIGNMENT));
  m_block_bitmap = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_tot_blocks, ALIGNMENT));

  if (!m_data_buf || !m_parity_buf || !m_block_bitmap) throw_error("XorecBenchmark: Failed to allocate memory.");

  memset(m_block_bitmap, 1, m_num_tot_blocks);

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < m_num_data_blocks; ++i) {
    if (write_validation_pattern(i, m_data_buf+i*m_block_size, m_block_size)) throw_error("XorecBenchmark: Failed to write validation pattern");
  }
}

XorecBenchmark::~XorecBenchmark() noexcept {
  _mm_free(m_data_buf);
  _mm_free(m_parity_buf);
  _mm_free(m_block_bitmap);
}

int XorecBenchmark::encode() noexcept {
  XorecResult res = xorec_encode(m_data_buf, m_parity_buf, m_block_size, m_num_data_blocks, m_num_parity_blocks, m_version);
  return (res == XorecResult::Success) ? 0 : -1;
}

int XorecBenchmark::decode() noexcept {
  XorecResult res = xorec_decode(m_data_buf, m_parity_buf, m_block_size, m_num_data_blocks, m_num_parity_blocks, m_block_bitmap, m_version);
  return (res == XorecResult::Success) ? 0 : -1;
}


void XorecBenchmark::simulate_data_loss() noexcept {
  select_lost_block_idxs(m_num_data_blocks, m_num_parity_blocks, m_num_lost_blocks, m_block_bitmap);
  unsigned i;
  for (i = 0; i < m_num_data_blocks; ++i) {
    if (!m_block_bitmap[i]) memset(m_data_buf + i * m_block_size, 0, m_block_size);
  }

  for (; i < m_num_tot_blocks; ++i) {
    auto idx = i - m_num_data_blocks;
    if (!m_block_bitmap[i]) memset(m_parity_buf + idx * m_block_size, 0, m_block_size);
  }
}