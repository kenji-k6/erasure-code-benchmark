/**
 * @file wirehair_bm.cpp
 * @brief Benchmark implementation for the Wirehair EC library
 * 
 * Documentation can be found in wirehair_bm.h and abstract_bm.h
 */


#include "wirehair_bm.hpp"
#include "utils.hpp"


WirehairBenchmark::WirehairBenchmark(const BenchmarkConfig& config) noexcept : ECBenchmark(config) {
  if (wirehair_init()) throw_error("Wirehair: Initialization failed.");

  // Allocate buffers
  // Allocate buffers
  m_data_buf = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_data_blocks*m_block_size, ALIGNMENT));
  m_encode_buf = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_tot_blocks*m_block_size, ALIGNMENT));
  m_block_bitmap = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_tot_blocks, ALIGNMENT));

  
  if (!m_data_buf || !m_encode_buf || !m_block_bitmap) throw_error("Wirehair: Failed to allocate memory.");
  memset(m_block_bitmap, 1, m_num_tot_blocks);

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < m_num_data_blocks; ++i) {
    if (write_validation_pattern(i, m_data_buf+i*m_block_size, m_block_size)) throw_error("Wirehair: Failed to write validation pattern.");
  }
}

WirehairBenchmark::~WirehairBenchmark() noexcept {
  _mm_free(m_data_buf);
  _mm_free(m_encode_buf);
  _mm_free(m_block_bitmap);
}

int WirehairBenchmark::encode() noexcept {
  uint32_t write_len = 0;
  m_encoder = wirehair_encoder_create(m_encoder, m_data_buf, m_num_data_blocks*m_block_size, m_block_size);
  if (!m_encoder) return 1;
  for (size_t i = 0; i < m_num_tot_blocks; ++i) {
    if (wirehair_encode(m_encoder, i, m_encode_buf+i*m_block_size,
                        m_block_size, &write_len) != Wirehair_Success) return -1;
  }
  wirehair_free(m_encoder);
  m_encoder = nullptr;
  return 0;
}


int WirehairBenchmark::decode() noexcept {
  WirehairResult result = Wirehair_NeedMore;
  m_decoder = wirehair_decoder_create(m_decoder, m_block_size*m_num_data_blocks, m_block_size);
  if (!m_decoder) return 1;

  for (unsigned i = 0; i < m_num_tot_blocks; ++i) {
    if (!m_block_bitmap[i]) continue;
    result = wirehair_decode(m_decoder, i, m_encode_buf+i*m_block_size, m_block_size);
    if (result == Wirehair_Success) break;
  }

  result = wirehair_recover(m_decoder, m_data_buf, m_block_size*m_num_data_blocks);
  wirehair_free(m_decoder);
  return result;
}


void WirehairBenchmark::simulate_data_loss() noexcept {
  select_lost_block_idxs(m_num_data_blocks, m_num_parity_blocks, m_num_lost_blocks, m_block_bitmap);
  for (unsigned i = 0; i < m_num_tot_blocks; ++i) {
    if (!m_block_bitmap[i]) memset(m_encode_buf + i * m_block_size, 0, m_block_size);
  }
}