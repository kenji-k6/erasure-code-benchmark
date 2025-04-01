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
  if (xorec_encode(m_data_buf, m_parity_buf, m_block_size, m_num_data_blocks, m_num_parity_blocks, m_version) != XorecResult::Success)
    return -1;
  return 0;
}

int XorecBenchmark::decode() noexcept {
  if (xorec_decode(m_data_buf, m_parity_buf, m_block_size, m_num_data_blocks, m_num_parity_blocks, m_block_bitmap, m_version) != XorecResult::Success)
    return -1;
  return 0;
}


void XorecBenchmark::simulate_data_loss() noexcept {
  // unsigned loss_idx = 0;
  // for (unsigned i = 0; i < m_num_total_blocks; ++i) {
  //   if (loss_idx < m_num_lost_blocks && m_lost_block_idxs[loss_idx] == i) {
  //     if (i < m_num_original_blocks) {
  //       memset(&m_data_buffer[i * m_block_size], 0, m_block_size);
  //       m_block_bitmap[i] = 0;
  //     } else {
  //       memset(&m_parity_buffer[(i-m_num_original_blocks) * m_block_size], 0, m_block_size);
  //       m_block_bitmap[i-m_num_original_blocks + XOREC_MAX_DATA_BLOCKS] = 0;
  //     }

  //     ++loss_idx;
  //     continue;
  //   }
  //   if (i < m_num_original_blocks) {
  //     m_block_bitmap[i] = 1;
  //   } else {
  //     m_block_bitmap[i-m_num_original_blocks + XOREC_MAX_DATA_BLOCKS] = 1;
  //   }
  // }
}