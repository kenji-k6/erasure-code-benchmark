#include "abstract_bm.hpp"

AbstractBenchmark::AbstractBenchmark(const BenchmarkConfig& config) noexcept
  : m_block_size(config.block_size),
    m_num_data_blocks(get<1>(config.ec_params)),
    m_num_parity_blocks(get<0>(config.ec_params)-m_num_data_blocks),
    m_num_tot_blocks(m_num_data_blocks + m_num_parity_blocks),
    m_num_lost_blocks(config.num_lost_blocks),
    m_data_buf(make_unique_aligned<uint8_t>(m_block_size * m_num_tot_blocks)),
    m_parity_buf(make_unique_aligned<uint8_t>(m_block_size * m_num_parity_blocks)),
    m_block_bitmap(make_unique_aligned<uint8_t>(m_num_tot_blocks))
{}

void AbstractBenchmark::simulate_data_loss() noexcept {
  select_lost_blocks(m_num_data_blocks, m_num_parity_blocks, m_num_lost_blocks, m_block_bitmap.get());

  unsigned i;
  for (i = 0; i < m_num_data_blocks; ++i) {
    if (!m_block_bitmap[i]) memset(&m_data_buf[i*m_block_size], 0, m_block_size);
  }

  for (; i < m_num_tot_blocks; ++i) {
    auto idx = i - m_num_data_blocks;
    if (!m_block_bitmap[i]) memset(&m_parity_buf[idx*m_block_size], 0, m_block_size);
  }
}

bool AbstractBenchmark::check_for_corruption() const noexcept {
  for (unsigned i = 0; i < m_num_data_blocks; ++i) {
    if (!validate_block(&m_data_buf[i * m_block_size], m_block_size)) return false;
  }
  return true;
}

void AbstractBenchmark::m_write_data_buffer() noexcept {
  for (unsigned i = 0; i < m_num_data_blocks; ++i) {
    if (write_validation_pattern(&m_data_buf[i*m_block_size], m_block_size)) {
      throw_error("Failed to write random checking packet.");
    }
  }
}