#include "abstract_bm.hpp"


AbstractBenchmark::AbstractBenchmark(const BenchmarkConfig& config) noexcept
  : m_threads(config.num_cpu_threads),
    m_message_size(config.message_size),
    m_block_size(config.block_size),
    m_chunk_data_blocks(get<1>(config.ec_params)),
    m_chunk_parity_blocks(get<0>(config.ec_params)-m_chunk_data_blocks),
    m_chunk_tot_blocks(m_chunk_data_blocks + m_chunk_parity_blocks),
    m_chunks(m_message_size / (m_block_size * m_chunk_data_blocks)),
    m_chunk_data_size(m_block_size * m_chunk_data_blocks),
    m_chunk_parity_size(m_block_size * m_chunk_parity_blocks),
    m_chunk_lost_blocks(config.num_lost_blocks),
    m_data_buf(make_unique_aligned<uint8_t>(m_chunks * m_chunk_data_size)),
    m_parity_buf(make_unique_aligned<uint8_t>(m_chunks * m_chunk_parity_size)),
    m_block_bitmap(make_unique_aligned<uint8_t>(m_chunks * m_chunk_tot_blocks))
{}

void AbstractBenchmark::simulate_data_loss() noexcept {

  for (unsigned c = 0; c < m_chunks; ++c) {
    auto bitmap = m_block_bitmap.get() + c * m_chunk_tot_blocks;
    auto data_buf = m_data_buf.get() + c * m_chunk_data_size;
    auto parity_buf = m_parity_buf.get() + c * m_chunk_parity_size;

    select_lost_blocks(m_chunk_data_blocks, m_chunk_parity_blocks, m_chunk_lost_blocks, bitmap);

    unsigned i;
    for (i = 0; i < m_chunk_data_blocks; ++i) {
      if (!bitmap[i]) memset(&data_buf[i*m_block_size], 0, m_block_size);
    }

    for (; i < m_chunk_tot_blocks; ++i) {
      auto idx = i - m_chunk_data_blocks;
      if (!bitmap[i]) memset(&parity_buf[idx*m_block_size], 0, m_block_size);
    }
  }
}

bool AbstractBenchmark::check_for_corruption() const noexcept {
  for (unsigned c = 0; c < m_chunks; ++c) {
    auto data_buf = m_data_buf.get() + c * m_chunk_data_size;

    for (unsigned i = 0; i < m_chunk_data_blocks; ++i) {
      if (!validate_block(&data_buf[i*m_block_size], m_block_size)) return false;
    }
  }
  return true;
}

void AbstractBenchmark::m_write_data_buffer() noexcept {
  for (unsigned c = 0; c < m_chunks; ++c) {
    auto data_buf = m_data_buf.get() + c * m_chunk_data_size;
    for (unsigned i = 0; i < m_chunk_data_blocks; ++i) {
      if (write_validation_pattern(&data_buf[i*m_block_size], m_block_size)) {
        throw_error("Failed to write random checking packet.");
      }
    }
  }
}