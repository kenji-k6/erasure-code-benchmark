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
  size_t num_data_blocks = get<0>(m_fec_params);
  size_t num_parity_blocks = get<1>(m_fec_params);

  // Initialize CM256 parameters
  m_params.BlockBytes = m_size_blk;
  m_params.OriginalCount = num_data_blocks;
  m_params.RecoveryCount = num_parity_blocks;

  // Allocate memory
  m_data_buffer = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_chunks * m_size_data_submsg, ALIGNMENT));
  m_parity_buffer = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_chunks * m_size_parity_submsg, ALIGNMENT));
  m_block_bitmap = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_chunks * m_blks_per_chunk, ALIGNMENT));

  if (!m_data_buffer || !m_parity_buffer || !m_block_bitmap) throw_error("CM256: Failed to allocate memory.");

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < m_size_msg/m_size_blk; ++i) {
    if (write_validation_pattern(i, m_data_buffer, m_size_blk)) throw_error("CM256: Failed to write validation pattern");
  }

  // Initialize indices
  for (unsigned i = 0; i < m_num_chunks; ++i) {
    for (unsigned idx = 0; idx < num_data_blocks; ++i) {
      m_blocks[i*num_data_blocks + idx].Index = cm256_get_original_block_index(m_params, idx);
    }
  }
  
  // Initialize block vector
  m_blocks.resize(ECLimits::CM256_MAX_TOT_BLOCKS*m_num_chunks);
  for (unsigned i = 0; i < m_size_msg/m_size_blk; ++i) {
    m_blocks[i].Block = m_data_buffer + i * m_size_blk;
  }

}

CM256Benchmark::~CM256Benchmark() noexcept {
  _mm_free(m_data_buffer);
  _mm_free(m_parity_buffer);
  _mm_free(m_block_bitmap);
}

int CM256Benchmark::encode() noexcept {
  for (unsigned i = 0; i < m_num_chunks; ++i) {
    uint8_t* parity_ptr = m_parity_buffer + i * m_size_parity_submsg;
    cm256_block* blocks = m_blocks.data() + i * ECLimits::CM256_MAX_TOT_BLOCKS;
    if (cm256_encode(m_params, blocks, parity_ptr)) return 1;
  }
  return 0;
}

int CM256Benchmark::decode() noexcept {
  for (unsigned i = 0; i < m_num_chunks; ++i) {
    cm256_block* blocks = m_blocks.data() + i * ECLimits::CM256_MAX_TOT_BLOCKS;
    if (cm256_decode(m_params, blocks)) return 1;
  }
  return 0;
}

void CM256Benchmark::simulate_data_loss() noexcept {
  // TODO
  return;
}
