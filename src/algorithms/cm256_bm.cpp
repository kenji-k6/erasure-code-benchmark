/**
 * @file cm256_bm.cpp
 * @brief Benchmark implementation for the CM256 EC Library
 * 
 * Documentation can be found in cm256_bm.h and abstract_bm.h
 */

 
#include "cm256_bm.hpp"
#include "utils.hpp"
#include <ranges>

CM256Benchmark::CM256Benchmark(const BenchmarkConfig& config) noexcept
  : AbstractBenchmark(config),
    m_params{
      .OriginalCount = static_cast<int>(m_chunk_data_blocks),
      .RecoveryCount = static_cast<int>(m_chunk_parity_blocks),
      .BlockBytes = static_cast<int>(m_block_size)
    }

{
  if (cm256_init()) throw_error("CM256: Initialization failed.");
  m_blocks.reserve(ECLimits::CM256_MAX_TOT_BLOCKS * m_chunks);
}

void CM256Benchmark::setup() noexcept {
  std::fill_n(m_block_bitmap.get(), m_chunks * m_chunk_tot_blocks, 1);

  // Initialize block vector
  for (unsigned c = 0; c < m_chunks; ++c) {
    auto blocks = m_blocks.data() + c * ECLimits::CM256_MAX_TOT_BLOCKS;
    auto data_buf = m_data_buf.get() + c * m_chunk_data_size;

    for (unsigned i = 0; i < m_chunk_data_blocks; ++i) {
      blocks[i] = {
        .Block = &data_buf[i*m_block_size],
        .Index = cm256_get_original_block_index(m_params, i)
      };
    }
  }
  
  m_write_data_buffer();
  omp_set_num_threads(m_threads);
}


int CM256Benchmark::encode() noexcept {
  int return_code = 0;

  #pragma omp parallel for
  for (unsigned c = 0; c < m_chunks; ++c) {
    auto blocks = m_blocks.data() + c * ECLimits::CM256_MAX_TOT_BLOCKS;
    auto parity_buf = m_parity_buf.get() + c * m_chunk_parity_size;

    if (cm256_encode(m_params, blocks, parity_buf)) {
      #pragma omp atomic write
      return_code = 1;
    }
  }

  return return_code;
}


int CM256Benchmark::decode() noexcept {
  int return_code = 0;

  #pragma omp parallel for
  for (unsigned c = 0; c < m_chunks; ++c) {
    auto blocks = m_blocks.data() + c * ECLimits::CM256_MAX_TOT_BLOCKS;
    auto bitmap = m_block_bitmap.get() + c * m_chunk_tot_blocks;
    auto data_buf = m_data_buf.get() + c * m_chunk_data_size;
    auto parity_buf = m_parity_buf.get() + c * m_chunk_parity_size;

    // replace the lost blocks with corresponding recovery block
    uint32_t recovery_idx = 0;
    for (unsigned i = 0; i < m_chunk_data_blocks; ++i) {
      if (!bitmap[i]) {
        while (recovery_idx < m_chunk_parity_blocks && !bitmap[m_chunk_data_blocks + recovery_idx]) {
          ++recovery_idx;
        }
        if (recovery_idx == m_chunk_parity_blocks) {
          #pragma omp atomic write
          return_code = 1;
        }
        blocks[i] = {
          .Block = &parity_buf[recovery_idx*m_block_size],
          .Index = cm256_get_recovery_block_index(m_params, recovery_idx)
        };
        ++recovery_idx;
      }
    }

    cm256_decode(m_params, blocks);
    for (unsigned i = 0; i < m_chunk_data_blocks; ++i) {
      if (!bitmap[i]) {
        memcpy(&data_buf[i*m_block_size], blocks[i].Block, m_block_size);
      }
    }
  }
  return return_code;
}
