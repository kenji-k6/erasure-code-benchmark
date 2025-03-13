/**
 * @file cm256_benchmark.cpp
 * @brief Benchmark implementation for the CM256 EC Library
 * 
 * Documentation can be found in cm256_benchmark.h and abstract_benchmark.h
 */

 
#include "cm256_benchmark.hpp"
#include "utils.hpp"
#include <iostream>
#include <memory>

CM256Benchmark::CM256Benchmark(const BenchmarkConfig& config) noexcept : ECBenchmark(config) {
  if (cm256_init()) throw_error("CM256: Initialization failed.");

  // Initialize CM256 parameters
  m_params.BlockBytes = m_block_size;
  m_params.OriginalCount = m_num_original_blocks;
  m_params.RecoveryCount = m_num_recovery_blocks;

  // Allocate buffers
  m_original_buffer = std::make_unique<uint8_t[]>(m_block_size * m_num_original_blocks);
  m_decode_buffer = std::make_unique<uint8_t[]>(m_block_size * m_num_recovery_blocks);

  if (!m_original_buffer || !m_decode_buffer) throw_error("CM256: Failed to allocate buffer(s).");

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < m_num_original_blocks; ++i) {
    int write_res = write_validation_pattern(i, &m_original_buffer[i * m_block_size], m_block_size);
    if (write_res) throw_error("CM256: Failed to write random checking packet.");
  }
  
  // Initialize block vector
  m_blocks.resize(ECLimits::CM256_MAX_TOT_BLOCKS);
  for (unsigned i = 0; i < m_num_original_blocks; ++i) {
    m_blocks[i].Block = &m_original_buffer[i * m_block_size];
  }
}


int CM256Benchmark::encode() noexcept {
  if (cm256_encode(m_params, m_blocks.data(), m_decode_buffer.get())) return 1;

  for (unsigned i = 0; i < m_num_original_blocks; ++i) {
    m_blocks[i].Index = cm256_get_original_block_index(m_params, i);
  }

  return 0;
}


int CM256Benchmark::decode() noexcept {
  return cm256_decode(m_params, m_blocks.data());
}


void CM256Benchmark::simulate_data_loss() noexcept {
  for (unsigned i = 0; i < m_num_lost_blocks; i++) {
    uint32_t idx = m_lost_block_idxs[i];

    if (idx < m_num_original_blocks) { // dropped block is original block
      idx = cm256_get_original_block_index(m_params, idx);
      memset(&m_original_buffer[idx * m_block_size], 0, m_block_size);
      m_blocks[idx].Block = &m_decode_buffer[i * m_block_size];
      m_blocks[idx].Index = cm256_get_recovery_block_index(m_params, i);

    } else { // dropped block is recovery block
      uint32_t orig_idx = idx - m_num_original_blocks;
      memset(&m_decode_buffer[orig_idx * m_block_size], 0, m_block_size);
    }
  }
}

bool CM256Benchmark::check_for_corruption() const noexcept {
  for (unsigned i = 0; i < m_num_original_blocks; i++) {
    if (!validate_block(static_cast<uint8_t*>(m_blocks[i].Block), m_block_size)) return false;
  }
  return true;
}
