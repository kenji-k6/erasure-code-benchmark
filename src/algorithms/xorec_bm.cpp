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
  m_num_total_blocks = m_num_original_blocks + m_num_recovery_blocks;
  m_data_buffer = std::make_unique<uint8_t[]>(m_block_size * m_num_original_blocks);
  m_parity_buffer = std::make_unique<uint8_t[]>(m_block_size * m_num_recovery_blocks);
  m_block_bitmap = std::make_unique<uint8_t[]>(XOREC_MAX_TOTAL_BLOCKS);
  m_version = config.xorec_params.version;


  if (!m_data_buffer || !m_parity_buffer) throw_error("Xorec: Failed to allocate buffer(s).");

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < m_num_original_blocks; ++i) {
    int write_res = write_validation_pattern(i, &m_data_buffer[i * m_block_size], m_block_size);
    if (write_res) throw_error("Xorec: Failed to write random checking packet.");
  }
}

int XorecBenchmark::encode() noexcept {
  xorec_encode(m_data_buffer.get(), m_parity_buffer.get(), m_block_size, m_num_original_blocks, m_num_recovery_blocks, m_version);
  return 0;
}

int XorecBenchmark::decode() noexcept {
  xorec_decode(m_data_buffer.get(), m_parity_buffer.get(), m_block_size, m_num_original_blocks, m_num_recovery_blocks, m_block_bitmap.get(), m_version);
  return 0;
}


void XorecBenchmark::simulate_data_loss() noexcept {
  unsigned loss_idx = 0;
  for (unsigned i = 0; i < m_num_total_blocks; ++i) {
    if (loss_idx < m_num_lost_blocks && m_lost_block_idxs[loss_idx] == i) {
      if (i < m_num_original_blocks) {
        memset(&m_data_buffer[i * m_block_size], 0, m_block_size);
        m_block_bitmap[i] = 0;
      } else {
        memset(&m_parity_buffer[(i-m_num_original_blocks) * m_block_size], 0, m_block_size);
        m_block_bitmap[i-m_num_original_blocks + XOREC_MAX_DATA_BLOCKS] = 0;
      }

      ++loss_idx;
      continue;
    }
    if (i < m_num_original_blocks) {
      m_block_bitmap[i] = 1;
    } else {
      m_block_bitmap[i-m_num_original_blocks + XOREC_MAX_DATA_BLOCKS] = 1;
    }
  }
}

bool XorecBenchmark::check_for_corruption() const noexcept {
  for (unsigned i = 0; i < m_num_original_blocks; ++i) {
    if (!validate_block(&m_data_buffer[i * m_block_size], m_block_size)) return false;
  }
  return true;
}