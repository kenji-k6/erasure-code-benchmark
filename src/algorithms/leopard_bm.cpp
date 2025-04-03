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
  memset(m_block_bitmap, 1, m_num_chunks * m_blks_per_chunk);
  
  if (!m_data_buffer || !m_encode_buffer || !m_decode_buffer || !m_block_bitmap) throw_error("Leopard: Failed to allocate memory.");

  // Populate vectors with pointers to the data blocks
  m_original_ptrs.reserve(m_num_chunks * m_data_blks_per_chunk);
  m_encode_work_ptrs.reserve(m_num_chunks * m_encode_work_count);
  m_decode_work_ptrs.reserve(m_num_chunks * m_decode_work_count);

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
  int return_code = 0;

  #pragma omp parallel for
  for (unsigned i = 0; i < m_num_chunks; ++i) {
    void** data_ptrs = reinterpret_cast<void**>(m_original_ptrs.data()) + i * m_data_blks_per_chunk;
    void** encode_work_ptrs = reinterpret_cast<void**>(m_encode_work_ptrs.data()) + i * m_encode_work_count;

    if (leo_encode(m_size_blk, m_data_blks_per_chunk, m_parity_blks_per_chunk, m_encode_work_count, data_ptrs, encode_work_ptrs)) {
      #pragma omp atomic write
      return_code = 1;
    }
  }

  return return_code;
}


int LeopardBenchmark::decode() noexcept {
  int return_code = 0;

  #pragma omp parallel for
  for (unsigned c = 0; c < m_num_chunks; ++c) {
    uint8_t* data_buf = m_data_buffer + c * m_size_data_submsg;
    uint8_t* bitmap = m_block_bitmap + c * m_blks_per_chunk;
    void** orig_ptrs = reinterpret_cast<void**>(m_original_ptrs.data()) + c * m_data_blks_per_chunk;
    void** encode_work_ptrs = reinterpret_cast<void**>(m_encode_work_ptrs.data()) + c * m_encode_work_count;
    void** decode_work_ptrs = reinterpret_cast<void**>(m_decode_work_ptrs.data()) + c * m_decode_work_count;

    unsigned i;
    for (i = 0; i < m_data_blks_per_chunk; ++i) {
      if (!bitmap[i]) orig_ptrs[i] = nullptr;
    }

    for (; i < m_blks_per_chunk; ++i) {
      auto idx = i - m_data_blks_per_chunk;
      if (!bitmap[i]) encode_work_ptrs[idx] = nullptr;
    }

    if (leo_decode(m_size_blk, m_data_blks_per_chunk, m_parity_blks_per_chunk,
                    m_decode_work_count, orig_ptrs, encode_work_ptrs, decode_work_ptrs)) {
        #pragma omp atomic write
        return_code = 1;
    }

    for (i = 0; i < m_data_blks_per_chunk; ++i) {
      if (!bitmap[i]) memcpy(data_buf + i * m_size_blk, decode_work_ptrs[i], m_size_blk);
    }
  }
  return return_code;
}


void LeopardBenchmark::simulate_data_loss() noexcept {
  for (unsigned c = 0; c < m_num_chunks; ++c) {
    uint8_t* bitmap = m_block_bitmap + c * m_blks_per_chunk;
    uint8_t* data_buf = m_data_buffer + c * m_size_data_submsg;
    uint8_t* encode_buf = m_encode_buffer + c * m_size_blk * m_encode_work_count;

    select_lost_block_idxs(m_data_blks_per_chunk, m_parity_blks_per_chunk, m_num_lst_rdma_pkts, bitmap);
    unsigned i;

    for (i = 0; i < m_data_blks_per_chunk; ++i) {
      if (!bitmap[i]) memset(data_buf + i * m_size_blk, 0, m_size_blk);
    }
  
    for (; i < m_blks_per_chunk; ++i) {
      auto idx = i - m_data_blks_per_chunk;
      if (!bitmap[i]) memset(encode_buf + idx * m_size_blk, 0, m_size_blk);
    }
  }
}