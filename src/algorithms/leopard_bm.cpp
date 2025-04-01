/**
 * @file leopard_bm.cpp
 * @brief Benchmark implementation for the Leopard EC library
 * 
 * Documentation can be found in leopard_bm.h and abstract_bm.h
 */


#include "leopard_bm.hpp"

#include "leopard.h"
#include "utils.hpp"


LeopardBenchmark::LeopardBenchmark(const BenchmarkConfig& config) noexcept : ECBenchmark(config) {
  // Initialize Leopard
  if (leo_init()) throw_error("Leopard: Initialization failed.");

  m_encode_work_count = leo_encode_work_count(m_num_data_blocks, m_num_parity_blocks);
  m_decode_work_count = leo_decode_work_count(m_num_data_blocks, m_num_parity_blocks);

  if (m_encode_work_count == 0 || m_decode_work_count == 0) throw_error("Leopard: Invalid work count(s).");

  // Allocate buffers

  m_data_buf = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_data_blocks*m_block_size, ALIGNMENT));
  m_encode_buf = reinterpret_cast<uint8_t*>(_mm_malloc(m_encode_work_count*m_block_size, ALIGNMENT));
  m_decode_buf = reinterpret_cast<uint8_t*>(_mm_malloc(m_decode_work_count*m_block_size, ALIGNMENT));
  m_block_bitmap = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_tot_blocks, ALIGNMENT));
  memset(m_block_bitmap, 1, m_num_tot_blocks);


  if (!m_data_buf || !m_encode_buf || !m_decode_buf || !m_block_bitmap) throw_error("Leopard: Failed to allocate memory.");

  // Populate vectors with pointers to the data blocks
  m_original_ptrs.reserve(m_num_data_blocks);
  m_encode_work_ptrs.reserve(m_encode_work_count);
  m_decode_work_ptrs.reserve(m_decode_work_count);

  for (unsigned i = 0; i < m_num_data_blocks; ++i) m_original_ptrs[i] = m_data_buf + i * m_block_size;
  for (unsigned i = 0; i < m_encode_work_count; ++i) m_encode_work_ptrs[i] = m_encode_buf + i * m_block_size;
  for (unsigned i = 0; i < m_decode_work_count; ++i) m_decode_work_ptrs[i] = m_decode_buf + i * m_block_size;

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < m_num_data_blocks; ++i) {
    if (write_validation_pattern(i, m_data_buf+i*m_block_size, m_block_size)) throw_error("Leopard: Failed to write random checking packet.");
  }
}

LeopardBenchmark::~LeopardBenchmark() noexcept {
  _mm_free(m_data_buf);
  _mm_free(m_encode_buf);
  _mm_free(m_decode_buf);
  _mm_free(m_block_bitmap);
}

int LeopardBenchmark::encode() noexcept {
  return leo_encode(m_block_size, m_num_data_blocks,
                    m_num_parity_blocks, m_encode_work_count,
                    reinterpret_cast<void**>(m_original_ptrs.data()),
                    reinterpret_cast<void**>(m_encode_work_ptrs.data()));
}


int LeopardBenchmark::decode() noexcept {
  if (leo_decode(m_block_size, m_num_data_blocks,
                    m_num_parity_blocks, m_decode_work_count,
                    reinterpret_cast<void**>(m_original_ptrs.data()),
                    reinterpret_cast<void**>(m_encode_work_ptrs.data()),
                    reinterpret_cast<void**>(m_decode_work_ptrs.data()))) 
  {
    return -1;
  }

  for (unsigned i = 0; i < m_num_data_blocks; ++i) {
    if (!m_original_ptrs[i]) memcpy(m_original_ptrs[i], m_decode_work_ptrs[i], m_block_size);
  }

  return 0;
}


void LeopardBenchmark::simulate_data_loss() noexcept {
  // for (auto idx : m_lost_block_idxs) {
  //   if (idx < m_num_data_blocks) {
  //     memset(m_original_ptrs[idx], 0, m_block_size);
  //     m_original_ptrs[idx] = nullptr;
  //   } else {
  //     memset(m_encode_work_ptrs[idx - m_num_data_blocks], 0, m_block_size);
  //     m_encode_work_ptrs[idx- m_num_data_blocks] = nullptr;
  //   }
  // }
}
