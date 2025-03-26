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
  m_params.BlockBytes = m_size_blk;
  m_params.OriginalCount = m_data_blks_per_chunk;
  m_params.RecoveryCount = m_parity_blks_per_chunk;

  // Allocate memory
  m_data_buffer = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_chunks * m_size_data_submsg, ALIGNMENT));
  m_parity_buffer = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_chunks * m_size_parity_submsg, ALIGNMENT));
  m_block_bitmap = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_chunks * m_blks_per_chunk, ALIGNMENT));
  memset(m_block_bitmap, 1, m_num_chunks * m_blks_per_chunk);
  if (!m_data_buffer || !m_parity_buffer || !m_block_bitmap) throw_error("CM256: Failed to allocate memory.");

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < m_size_msg/m_size_blk; ++i) {
    if (write_validation_pattern(i, m_data_buffer+i*m_size_blk, m_size_blk)) throw_error("CM256: Failed to write validation pattern");
  }


  m_blocks.resize(ECLimits::CM256_MAX_TOT_BLOCKS*m_num_chunks);

  // Initialize indices
  for (unsigned i = 0; i < m_num_chunks; ++i) {
    for (unsigned idx = 0; idx < m_data_blks_per_chunk; ++idx) {
      m_blocks[i*ECLimits::CM256_MAX_TOT_BLOCKS + idx].Index = cm256_get_original_block_index(m_params, idx);
    }
  }

  // Initialize block vector
  for (unsigned i = 0; i < m_num_chunks; ++i) {
    for (unsigned idx = 0; idx < m_data_blks_per_chunk; ++idx) {
      m_blocks[i*ECLimits::CM256_MAX_TOT_BLOCKS + idx].Block = m_data_buffer + i * m_size_data_submsg + idx * m_size_blk;
    }
  }

}

CM256Benchmark::~CM256Benchmark() noexcept {
  _mm_free(m_data_buffer);
  _mm_free(m_parity_buffer);
  _mm_free(m_block_bitmap);
}

int CM256Benchmark::encode() noexcept {
  cm256_block* blocks = m_blocks.data();
  uint8_t* parity_ptr = m_parity_buffer;

  for (unsigned i = 0; i < m_num_chunks; ++i) {
    if (cm256_encode(m_params, blocks, parity_ptr)) return 1;
    
    blocks += ECLimits::CM256_MAX_TOT_BLOCKS;
    parity_ptr += m_size_parity_submsg;
  }
  return 0;
}

int CM256Benchmark::decode() noexcept {
  cm256_block* blocks = m_blocks.data();
  uint8_t* bitmap = m_block_bitmap;

  for (unsigned i = 0; i < m_num_chunks; ++i) {
    if (cm256_decode(m_params, blocks)) return 1;
    blocks += ECLimits::CM256_MAX_TOT_BLOCKS;
    bitmap += m_blks_per_chunk;
  }

  return 0;
}

void CM256Benchmark::simulate_data_loss() noexcept {
  // TODO
  return;
}
