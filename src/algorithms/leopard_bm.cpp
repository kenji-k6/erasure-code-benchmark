/**
 * @file leopard_bm.cpp
 * @brief Benchmark implementation for the Leopard EC library
 * 
 * Documentation can be found in leopard_bm.h and abstract_bm.h
 */


#include "leopard_bm.hpp"

#include "leopard.h"
#include "utils.hpp"


LeopardBenchmark::LeopardBenchmark(const BenchmarkConfig& config) noexcept
  : AbstractBenchmark(config),
    m_parity_work_count(leo_encode_work_count(m_num_data_blocks, m_num_parity_blocks)),
    m_recovery_work_count(leo_decode_work_count(m_num_data_blocks, m_num_parity_blocks)),
    m_recovery_buf(make_unique_aligned<uint8_t>(m_block_size * m_recovery_work_count)),
    m_data_ptrs(m_num_data_blocks),
    m_parity_ptrs(m_parity_work_count),
    m_recovery_ptrs(m_recovery_work_count)
{
  // Overwrite default initialization
  m_parity_buf = make_unique_aligned<uint8_t>(m_block_size * m_parity_work_count);

  // Initialize Leopard
  if (leo_init() || m_parity_work_count == 0 || m_recovery_work_count == 0) {
    throw_error("Leopard: Initialization failed.");
  }
}

void LeopardBenchmark::setup() noexcept {
  std::fill_n(m_block_bitmap.get(), m_num_tot_blocks, 1);

  for (unsigned i = 0; i < m_num_data_blocks; ++i) m_data_ptrs[i] = &m_data_buf[i*m_block_size];
  for (unsigned i = 0; i < m_parity_work_count; ++i) m_parity_ptrs[i] = &m_parity_buf[i*m_block_size];
  for (unsigned i = 0; i < m_recovery_work_count; ++i) m_recovery_ptrs[i] = &m_recovery_buf[i*m_block_size];
  
  m_write_data_buffer();
}

int LeopardBenchmark::encode() noexcept {
  return leo_encode(m_block_size, m_num_data_blocks,
                    m_num_parity_blocks, m_parity_work_count,
                    reinterpret_cast<void**>(m_data_ptrs.data()),
                    reinterpret_cast<void**>(m_parity_ptrs.data()));
}


int LeopardBenchmark::decode() noexcept {
  unsigned i;
  for (i = 0; i < m_num_data_blocks; ++i) {
    if (!m_block_bitmap[i]) m_data_ptrs[i] = nullptr;
  }

  for (; i < m_num_tot_blocks; ++i) {
    auto idx = i - m_num_data_blocks;
    if (!m_block_bitmap[i]) m_parity_ptrs[idx] = nullptr;
  }

  if (leo_decode(m_block_size, m_num_data_blocks,
                    m_num_parity_blocks, m_recovery_work_count,
                    reinterpret_cast<void**>(m_data_ptrs.data()),
                    reinterpret_cast<void**>(m_parity_ptrs.data()),
                    reinterpret_cast<void**>(m_recovery_ptrs.data()))) 
  {
    return -1;
  }

  for (i = 0; i < m_num_data_blocks; ++i) {
    if (!m_block_bitmap[i]) memcpy(&m_data_buf[i*m_block_size], m_recovery_ptrs[i], m_block_size);
  }

  return 0;
}
