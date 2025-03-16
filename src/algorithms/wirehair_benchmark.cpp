/**
 * @file wirehair_benchmark.cpp
 * @brief Benchmark implementation for the Wirehair EC library
 * 
 * Documentation can be found in wirehair_benchmark.h and abstract_benchmark.h
 */


#include "wirehair_benchmark.hpp"

#include "utils.hpp"
#include <cstring>
#include <iostream>


WirehairBenchmark::WirehairBenchmark(const BenchmarkConfig& config) noexcept : ECBenchmark(config) {
  m_num_total_blocks = m_num_original_blocks + m_num_recovery_blocks;
  if (wirehair_init()) throw_error("Wirehair: Initialization failed.");

  // Allocate buffers
  m_original_buffer = std::make_unique<uint8_t[]>(m_block_size * m_num_original_blocks);
  m_encode_buffer = std::make_unique<uint8_t[]>(m_block_size * m_num_total_blocks);
  m_decode_buffer = std::make_unique<uint8_t[]>(m_block_size * m_num_original_blocks);

  if (!m_original_buffer || !m_encode_buffer || !m_decode_buffer) throw_error("Wirehair: Failed to allocate buffer(s).");

  m_decoder = wirehair_decoder_create(nullptr, m_block_size * m_num_original_blocks, m_block_size);
  if (!m_decoder) throw_error("Wirehair: Failed to create decoder instance.");

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < m_num_original_blocks; ++i) {
    int write_res = write_validation_pattern(i, &m_original_buffer[i * m_block_size], m_block_size);
    if (write_res) throw_error("Wirehair: Failed to write random checking packet.");
  }
}

WirehairBenchmark::~WirehairBenchmark() noexcept {
  if (m_encoder) wirehair_free(m_encoder);
  if (m_decoder) wirehair_free(m_decoder);
}

int WirehairBenchmark::encode() noexcept {
  m_encoder = wirehair_encoder_create(nullptr, m_original_buffer.get(), m_num_original_blocks * m_block_size, m_block_size);
  if (!m_encoder) return -1;
  uint32_t write_len = 0;
  for (size_t i = 0; i < m_num_original_blocks + m_num_recovery_blocks; ++i) {
    if (wirehair_encode(m_encoder, i, &m_encode_buffer[i * m_block_size],
                        m_block_size, &write_len) != Wirehair_Success) return -1;
  }
  return 0;
}


int WirehairBenchmark::decode() noexcept {
  WirehairResult decode_result = Wirehair_NeedMore;
  unsigned loss_idx = 0;

  for (unsigned i = 0; i < m_num_total_blocks; ++i) {
    if (loss_idx < m_num_lost_blocks && i == m_lost_block_idxs[loss_idx]) {
      loss_idx++;
      continue;
    }

    decode_result = wirehair_decode(m_decoder, i, &m_encode_buffer[i * m_block_size], m_block_size);
    if (decode_result == Wirehair_Success) break;
  }

  return wirehair_recover(m_decoder, m_decode_buffer.get(), m_block_size * m_num_original_blocks);
}


void WirehairBenchmark::simulate_data_loss() noexcept {
  for (auto idx : m_lost_block_idxs) {
    memset(&m_encode_buffer[idx * m_block_size], 0, m_block_size);
  }
}


bool WirehairBenchmark::check_for_corruption() const noexcept {
  for (unsigned i = 0; i < m_num_original_blocks; ++i) {
    if (!validate_block(&m_decode_buffer[i * m_block_size], m_block_size)) return false;
  }
  return true;
}