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
  if (!m_data_buffer || !m_parity_buffer || !m_block_bitmap) throw_error("CM256: Failed to allocate memory.");
  memset(m_block_bitmap, 1, m_num_chunks * m_blks_per_chunk);

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
  int return_code = 0;

  #pragma omp parallel for
  for (unsigned i = 0; i < m_num_chunks; ++i) {
    cm256_block* blocks = m_blocks.data() + i * ECLimits::CM256_MAX_TOT_BLOCKS;
    uint8_t* parity_ptr = m_parity_buffer + i * m_size_parity_submsg;

    if (cm256_encode(m_params, blocks, parity_ptr)) {
      #pragma omp atomic write
      return_code = 1;
    }
  }
  return return_code;
}

int CM256Benchmark::decode() noexcept {
  int return_code = 0;
  
  #pragma omp parallel for
  for (unsigned c = 0; c < m_num_chunks; ++c) {
    uint8_t* data_buf = m_data_buffer + c * m_size_data_submsg;
    uint8_t* parity_buf = m_parity_buffer + c * m_size_parity_submsg;
    cm256_block* blocks = m_blocks.data() + c * ECLimits::CM256_MAX_TOT_BLOCKS;
    uint8_t* bitmap = m_block_bitmap + c * m_blks_per_chunk;
    uint32_t recovery_idx = 0;

    for (unsigned i = 0; i < m_data_blks_per_chunk; ++i) {
      if (!bitmap[i]) {
        while (recovery_idx < m_parity_blks_per_chunk && !bitmap[m_data_blks_per_chunk + recovery_idx]) {
          ++recovery_idx;
        }

        if (recovery_idx == m_parity_blks_per_chunk) {
          #pragma omp atomic write
          return_code = 1;
          break;
        }

        blocks[i].Index = cm256_get_recovery_block_index(m_params, recovery_idx);
        blocks[i].Block = parity_buf + recovery_idx * m_size_blk;
        ++recovery_idx;
      }
    }

    cm256_decode(m_params, blocks);

    for (unsigned i = 0; i < m_data_blks_per_chunk; ++i) {
      if (!bitmap[i]) {
        memcpy(data_buf + i * m_size_blk, blocks[i].Block, m_size_blk);
      }
    }
  }
  return return_code;
}

void CM256Benchmark::simulate_data_loss() noexcept {
  for (unsigned c = 0; c < m_num_chunks; ++c) {
    uint8_t* bitmap = m_block_bitmap + c * m_blks_per_chunk;
    uint8_t* data_buf = m_data_buffer + c * m_size_data_submsg;
    uint8_t* parity_buf = m_parity_buffer + c * m_size_parity_submsg;

    select_lost_block_idxs(m_data_blks_per_chunk, m_parity_blks_per_chunk, m_num_lst_rdma_pkts, bitmap);
    unsigned i;

    for (i = 0; i < m_data_blks_per_chunk; ++i) {
      if (!bitmap[i]) memset(data_buf + i * m_size_blk, 0, m_size_blk);
    }

    for (; i < m_blks_per_chunk; ++i) {
      if (!bitmap[i]) memset(parity_buf + (i - m_data_blks_per_chunk) * m_size_blk, 0, m_size_blk);
    }
  }
  return;
}
