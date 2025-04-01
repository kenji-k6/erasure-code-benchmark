#include "abstract_bm.hpp"

ECBenchmark::ECBenchmark(const BenchmarkConfig& config) noexcept
  : m_block_size(config.block_size),
    m_num_data_blocks(config.num_data_blocks),
    m_num_parity_blocks(config.num_parity_blocks),
    m_num_tot_blocks(config.num_data_blocks + config.num_parity_blocks),
    m_num_lost_blocks(config.num_lost_blocks)
{}

bool ECBenchmark::check_for_corruption() const noexcept {
  for (unsigned i = 0; i < m_num_data_blocks; ++i) {
    if (!validate_block(m_data_buf+i*m_block_size, m_block_size)) return false;
  }
  return true;
}