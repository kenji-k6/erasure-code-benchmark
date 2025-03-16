/**
 * @file leopard_benchmark.cpp
 * @brief Benchmark implementation for the Leopard EC library
 * 
 * Documentation can be found in leopard_benchmark.h and abstract_benchmark.h
 */


#include "leopard_benchmark.hpp"

#include "leopard.h"
#include "utils.hpp"
#include <cstring>
#include <iostream>


LeopardBenchmark::LeopardBenchmark(const BenchmarkConfig& config) noexcept : ECBenchmark(config) {
  // Initialize Leopard
  if (leo_init()) throw_error("Leopard: Initialization failed.");

  m_encode_work_count = leo_encode_work_count(m_num_original_blocks, m_num_recovery_blocks);
  m_decode_work_count = leo_decode_work_count(m_num_original_blocks, m_num_recovery_blocks);

  if (m_encode_work_count == 0 || m_decode_work_count == 0) throw_error("Leopard: Invalid work count(s).");

  // Allocate buffers
  m_original_buffer = std::make_unique<uint8_t[]>(m_block_size * m_num_original_blocks);
  m_encode_buffer = std::make_unique<uint8_t[]>(m_block_size * m_encode_work_count);
  m_decode_buffer = std::make_unique<uint8_t[]>(m_block_size * m_decode_work_count);

  if (!m_original_buffer || !m_encode_buffer || !m_decode_buffer) throw_error("Leopard: Failed to allocate buffer(s).");

  // Populate vectors with pointers to the data blocks
  m_original_ptrs.resize(m_num_original_blocks);
  m_encode_work_ptrs.resize(m_encode_work_count);
  m_decode_work_ptrs.resize(m_decode_work_count);

  for (unsigned i = 0; i < m_num_original_blocks; ++i) m_original_ptrs[i] = &m_original_buffer[i * m_block_size];
  for (unsigned i = 0; i < m_encode_work_count; ++i) m_encode_work_ptrs[i] = &m_encode_buffer[i * m_block_size];
  for (unsigned i = 0; i < m_decode_work_count; ++i) m_decode_work_ptrs[i] = &m_decode_buffer[i * m_block_size];

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < m_num_original_blocks; ++i) {
    int write_res = write_validation_pattern(i, m_original_ptrs[i], m_block_size);
    if (write_res) throw_error("Leopard: Failed to write random checking packet.");
  }
}

int LeopardBenchmark::encode() noexcept {
  return leo_encode(m_block_size, m_num_original_blocks,
                    m_num_recovery_blocks, m_encode_work_count,
                    reinterpret_cast<void**>(m_original_ptrs.data()),
                    reinterpret_cast<void**>(m_encode_work_ptrs.data()));
}


int LeopardBenchmark::decode() noexcept {
  return leo_decode(m_block_size, m_num_original_blocks,
                    m_num_recovery_blocks, m_decode_work_count,
                    reinterpret_cast<void**>(m_original_ptrs.data()),
                    reinterpret_cast<void**>(m_encode_work_ptrs.data()),
                    reinterpret_cast<void**>(m_decode_work_ptrs.data()));
}


void LeopardBenchmark::simulate_data_loss() noexcept {
  for (auto idx : m_lost_block_idxs) {
    if (idx < m_num_original_blocks) {
      memset(m_original_ptrs[idx], 0, m_block_size);
      m_original_ptrs[idx] = nullptr;
    } else {
      memset(m_encode_work_ptrs[idx - m_num_original_blocks], 0, m_block_size);
      m_encode_work_ptrs[idx- m_num_original_blocks] = nullptr;
    }
  }
}


bool LeopardBenchmark::check_for_corruption() const noexcept {
  for (unsigned i = 0; i < m_num_original_blocks; ++i) {
    if (!m_original_ptrs[i]) { // lost block
      if (!validate_block(m_decode_work_ptrs[i], m_block_size)) return false;
    } else { // block is intact
      if (!validate_block(m_original_ptrs[i], m_block_size)) return false;
    }
  }
  return true;
}
