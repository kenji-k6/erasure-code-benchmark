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

  m_encode_work_count = leo_encode_work_count(m_data_blks_per_chunk, m_parity_blks_per_chunk);
  m_decode_work_count = leo_decode_work_count(m_data_blks_per_chunk, m_parity_blks_per_chunk);

  if (m_encode_work_count == 0 || m_decode_work_count == 0) throw_error("Leopard: Invalid work count(s).");

  // Allocate buffers;
  m_data_buffer = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_chunks * m_size_data_submsg, ALIGNMENT));
  m_encode_buffer = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_chunks * m_size_blk * m_encode_work_count, ALIGNMENT));
  m_decode_buffer = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_chunks * m_size_blk * m_decode_work_count, ALIGNMENT));
  m_block_bitmap = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_chunks * m_blks_per_chunk, ALIGNMENT));

  if (!m_data_buffer || !m_encode_buffer || !m_decode_buffer || !m_block_bitmap) throw_error("Leopard: Failed to allocate memory.");

  // Populate vectors with pointers to the data blocks
  m_original_ptrs.resize(m_num_chunks * m_data_blks_per_chunk);
  m_encode_work_ptrs.resize(m_num_chunks * m_encode_work_count);
  m_decode_work_ptrs.resize(m_num_chunks * m_decode_work_count);

  for (unsigned i = 0; i < m_num_chunks * m_data_blks_per_chunk; ++i) m_original_ptrs[i] = m_data_buffer + i*m_size_blk;
  for (unsigned i = 0; i < m_num_chunks * m_encode_work_count; ++i) m_encode_work_ptrs[i] = m_encode_buffer + i*m_size_blk;
  for (unsigned i = 0; i < m_num_chunks * m_decode_work_count; ++i) m_decode_work_ptrs[i] = m_decode_buffer + i*m_size_blk;

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < m_size_msg/m_size_blk; ++i) {
    if (write_validation_pattern(i, m_data_buffer+i*m_size_blk, m_size_blk)) throw_error("Leopard: Failed to write validation pattern.");
  }
}

LeopardBenchmark::~LeopardBenchmark() noexcept {
  _mm_free(m_data_buffer);
  _mm_free(m_encode_buffer);
  _mm_free(m_decode_buffer);
  _mm_free(m_block_bitmap);
}

int LeopardBenchmark::encode() noexcept {
  void** data_ptrs = reinterpret_cast<void**>(m_original_ptrs.data());
  void** encode_work_ptrs = reinterpret_cast<void**>(m_encode_work_ptrs.data());

  for (unsigned i = 0; i < m_num_chunks; ++i) {
    if (leo_encode(m_size_blk, m_data_blks_per_chunk, m_parity_blks_per_chunk, m_encode_work_count, data_ptrs, encode_work_ptrs)) return 1;
    data_ptrs += m_data_blks_per_chunk;
    encode_work_ptrs += m_encode_work_count;
  }

  return 0;
}


int LeopardBenchmark::decode() noexcept {
  void** data_ptrs = reinterpret_cast<void**>(m_original_ptrs.data());
  void** encode_work_ptrs = reinterpret_cast<void**>(m_encode_work_ptrs.data());
  void** decode_work_ptrs = reinterpret_cast<void**>(m_decode_work_ptrs.data());

  for (unsigned i = 0; i < m_num_chunks; ++i) {

    if (leo_decode(m_size_blk, m_data_blks_per_chunk, m_parity_blks_per_chunk, m_decode_work_count, data_ptrs, encode_work_ptrs, decode_work_ptrs)) return 1;

    for (unsigned j = 0; j < m_data_blks_per_chunk; ++j) {
      if (!data_ptrs[j]) memcpy(data_ptrs + j, decode_work_ptrs + j, m_size_blk);
    }

    data_ptrs += m_data_blks_per_chunk;
    encode_work_ptrs += m_encode_work_count;
    decode_work_ptrs += m_decode_work_count;
  }
  return 0;
}


void LeopardBenchmark::simulate_data_loss() noexcept {
  return; // TODO
}