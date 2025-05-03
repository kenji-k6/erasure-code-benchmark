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
    m_parity_work_count(leo_encode_work_count(m_chunk_data_blocks, m_chunk_parity_blocks)),
    m_recovery_work_count(leo_decode_work_count(m_chunk_data_blocks, m_chunk_parity_blocks)),
    m_recovery_buf(make_unique_aligned<uint8_t>(m_chunks * m_block_size * m_recovery_work_count))
{
  // Overwrite default initialization
  m_parity_buf = make_unique_aligned<uint8_t>(m_chunks * m_block_size * m_parity_work_count);

  m_data_ptrs.reserve(m_chunks * m_chunk_data_blocks);
  m_parity_ptrs.reserve(m_chunks * m_parity_work_count);
  m_recovery_ptrs.reserve(m_chunks * m_recovery_work_count);

  // Initialize Leopard
  if (leo_init() || m_parity_work_count == 0 || m_recovery_work_count == 0) {
    throw_error("Leopard: Initialization failed.");
  }
}

void LeopardBenchmark::setup() noexcept {
  std::fill_n(m_block_bitmap.get(), m_chunks * m_chunk_tot_blocks, 1);

  for (unsigned c = 0; c < m_chunks; ++c) {
    auto data_buf = m_data_buf.get() + c * m_chunk_data_size;
    auto parity_buf = m_parity_buf.get() + c * m_block_size * m_parity_work_count;
    auto recovery_buf = m_recovery_buf.get() + c * m_block_size * m_recovery_work_count;
    auto data_ptrs = m_data_ptrs.data() + c * m_chunk_data_blocks;
    auto parity_ptrs = m_parity_ptrs.data() + c * m_parity_work_count;
    auto recovery_ptrs = m_recovery_ptrs.data() + c * m_recovery_work_count;

    for (unsigned i = 0; i < m_chunk_data_blocks; ++i) data_ptrs[i] = &data_buf[i*m_block_size];
    for (unsigned i = 0; i < m_parity_work_count; ++i) parity_ptrs[i] = &parity_buf[i*m_block_size];
    for (unsigned i = 0; i < m_recovery_work_count; ++i) recovery_ptrs[i] = &recovery_buf[i*m_block_size];
  }

  m_write_data_buffer();
  omp_set_num_threads(m_threads);
}

int LeopardBenchmark::encode() noexcept {
  int return_code = 0;

  #pragma omp parallel for
  for (unsigned c = 0; c < m_chunks; ++c) {
    void** data_ptrs = reinterpret_cast<void**>(m_data_ptrs.data() + c * m_chunk_data_blocks);
    void** parity_ptrs = reinterpret_cast<void**>(m_parity_ptrs.data() + c * m_parity_work_count);

    if (leo_encode(m_block_size, m_chunk_data_blocks,
                   m_chunk_parity_blocks, m_parity_work_count,
                   data_ptrs, parity_ptrs)) {
      #pragma omp atomic write
      return_code = 1;
    }
  }
  return return_code;
}


int LeopardBenchmark::decode() noexcept {
  int return_code = 0;

  #pragma omp parallel for
  for (unsigned c = 0; c < m_chunks; ++c) {
    auto data_buf = m_data_buf.get() + c * m_chunk_data_size;
    auto bitmap = m_block_bitmap.get() + c * m_chunk_tot_blocks;
    void** data_ptrs = reinterpret_cast<void**>(m_data_ptrs.data() + c * m_chunk_data_blocks);
    void** parity_ptrs = reinterpret_cast<void**>(m_parity_ptrs.data() + c * m_parity_work_count);
    void** recovery_ptrs = reinterpret_cast<void**>(m_recovery_ptrs.data() + c * m_recovery_work_count);

    unsigned i;
    for (i = 0; i < m_chunk_data_blocks; ++i) {
      if (!bitmap[i]) data_ptrs[i] = nullptr;
    }

    for (; i < m_chunk_tot_blocks; ++i) {
      auto idx = i - m_chunk_data_blocks;
      if (!bitmap[i]) parity_ptrs[idx] = nullptr;
    }

    if (leo_decode(m_block_size, m_chunk_data_blocks,
                      m_chunk_parity_blocks, m_recovery_work_count,
                      data_ptrs, parity_ptrs, recovery_ptrs)) {
      #pragma omp atomic write
      return_code = 1;
    }

    for (i = 0; i < m_chunk_data_blocks; ++i) {
      if (!bitmap[i]) memcpy(&data_buf[i*m_block_size], recovery_ptrs[i], m_block_size);
    }
  }
  return return_code;
}
