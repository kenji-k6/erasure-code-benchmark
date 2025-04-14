/**
 * @file cm256_bm.cpp
 * @brief Benchmark implementation for the CM256 EC Library
 * 
 * Documentation can be found in cm256_bm.h and abstract_bm.h
 */

 
#include "cm256_bm.hpp"
#include "utils.hpp"
#include <ranges>

CM256Benchmark::CM256Benchmark(const BenchmarkConfig& config) noexcept
  : AbstractBenchmark(config),
    m_params{
      .OriginalCount = static_cast<int>(m_num_data_blocks),
      .RecoveryCount = static_cast<int>(m_num_parity_blocks),
      .BlockBytes = static_cast<int>(m_block_size)
    }

{
  if (cm256_init()) throw_error("CM256: Initialization failed.");
  
  // Initialize block vector
  for (unsigned i = 0; i < m_num_data_blocks; ++i) {
    m_blocks[i] = {
      .Block = &m_data_buf[i*m_block_size],
      .Index = cm256_get_original_block_index(m_params, i)
    };
  }
  
  m_write_data_buffer();
}


int CM256Benchmark::encode() noexcept {
  return cm256_encode(m_params, m_blocks.data(), m_parity_buf.get());
}


int CM256Benchmark::decode() noexcept {
  // replace the lost blocks with corresponding recovery block
  uint32_t recovery_idx = 0;

  for (unsigned i = 0; i < m_num_data_blocks; ++i) {
    if (!m_block_bitmap[i]) {
      // lost a data block
      while (recovery_idx < m_num_parity_blocks && !m_block_bitmap[m_num_data_blocks + recovery_idx]) {
        ++recovery_idx;
      }
      if (recovery_idx == m_num_parity_blocks) return 1;
      
      m_blocks[i] = {
        .Block = &m_parity_buf[recovery_idx*m_block_size],
        .Index = cm256_get_recovery_block_index(m_params, recovery_idx)
      };
      ++recovery_idx;
    }
  }

  cm256_decode(m_params, m_blocks.data());

  for (unsigned i = 0; i < m_num_data_blocks; ++i) {
    if (!m_block_bitmap[i]) {
      memcpy(&m_data_buf[i*m_block_size], m_blocks[i].Block, m_block_size);
    }
  }
  return 0;
}
