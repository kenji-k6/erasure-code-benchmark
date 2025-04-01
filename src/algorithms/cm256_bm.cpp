/**
 * @file cm256_bm.cpp
 * @brief Benchmark implementation for the CM256 EC Library
 * 
 * Documentation can be found in cm256_bm.h and abstract_bm.h
 */

 
#include "cm256_bm.hpp"
#include "utils.hpp"

CM256Benchmark::CM256Benchmark(const BenchmarkConfig& config) noexcept : ECBenchmark(config) {
  if (cm256_init()) throw_error("CM256: Initialization failed.");

  // Initialize CM256 parameters
  m_params.BlockBytes = m_block_size;
  m_params.OriginalCount = m_num_data_blocks;
  m_params.RecoveryCount = m_num_parity_blocks;

  // Allocate buffers
  m_data_buf = reinterpret_cast<uint8_t*>(_mm_malloc(m_block_size * m_num_data_blocks, ALIGNMENT));
  m_parity_buf = reinterpret_cast<uint8_t*>(_mm_malloc(m_block_size * m_num_parity_blocks, ALIGNMENT));
  m_block_bitmap = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_tot_blocks, ALIGNMENT));
  if (!m_data_buf || !m_parity_buf || !m_block_bitmap) throw_error("CM256: Failed to allocate buffer(s).");

  memset(m_block_bitmap, 1, m_num_tot_blocks);

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < m_num_data_blocks; ++i) {
    if (write_validation_pattern(i, m_data_buf+i*m_block_size, m_block_size)) throw_error("CM256: Failed to write random checking packet.");
  }
  
  // Initialize block vector
  m_blocks.resize(ECLimits::CM256_MAX_TOT_BLOCKS);
  for (unsigned i = 0; i < m_num_data_blocks; ++i) {
    m_blocks[i].Index = cm256_get_original_block_index(m_params, i);
    m_blocks[i].Block = m_data_buf + i * m_block_size;
  }
}

CM256Benchmark::~CM256Benchmark() noexcept{
  _mm_free(m_data_buf);
  _mm_free(m_parity_buf);
  _mm_free(m_block_bitmap);
}


int CM256Benchmark::encode() noexcept {
  return cm256_encode(m_params, m_blocks.data(), m_parity_buf);
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
      if (recovery_idx == m_num_parity_blocks) return -1;

      m_blocks[i].Index = cm256_get_recovery_block_index(m_params, recovery_idx);
      m_blocks[i].Block = m_parity_buf + recovery_idx * m_block_size;
      ++recovery_idx;
    }
  }

  cm256_decode(m_params, m_blocks.data());

  for (unsigned i = 0; i < m_num_data_blocks; ++i) {
    if (!m_block_bitmap[i]) {
      memcpy(m_data_buf + i * m_block_size, m_blocks[i].Block, m_block_size);
    }
  }
  return 0;
}


void CM256Benchmark::simulate_data_loss() noexcept {
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
