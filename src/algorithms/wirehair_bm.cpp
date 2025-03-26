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

  m_num_total_blocks = m_data_blks_per_chunk + m_parity_blks_per_chunk;

  // Allocate buffers
  m_data_buffer = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_chunks * m_size_data_submsg, ALIGNMENT));
  m_encode_buffer = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_chunks * (m_size_parity_submsg + m_size_data_submsg), ALIGNMENT));
  m_block_bitmap = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_chunks * m_blks_per_chunk, ALIGNMENT));

  memset(m_block_bitmap, 1, m_num_chunks * m_blks_per_chunk);

  if (!m_data_buffer || !m_encode_buffer || !m_block_bitmap) throw_error("Wirehair: Failed to allocate memory.");

  m_encoder = nullptr;
  m_decoder = nullptr;

  for (unsigned i = 0; i < m_size_msg/m_size_blk; ++i) {
    if (write_validation_pattern(i, m_data_buffer+i*m_size_blk, m_size_blk)) throw_error("Wirehair: Failed to write validation pattern.");
  }
}

WirehairBenchmark::~WirehairBenchmark() noexcept {
  _mm_free(m_data_buffer);
  _mm_free(m_encode_buffer);
  _mm_free(m_block_bitmap);
  wirehair_free(m_encoder);
  wirehair_free(m_decoder);
}

int WirehairBenchmark::encode() noexcept {
  uint32_t write_len = 0;
  uint8_t* data_ptr = m_data_buffer;
  uint8_t* encode_ptr = m_encode_buffer;

  for (unsigned i = 0; i < m_num_chunks; ++i) {
    m_encoder = wirehair_encoder_create(m_encoder, data_ptr, m_size_data_submsg, m_size_blk);

    if (!m_encoder) return 1;

    for (unsigned j = 0; j < m_num_total_blocks; ++j) {

      if (wirehair_encode(m_encoder, j, encode_ptr + j * m_size_blk,
                          m_size_blk, &write_len) != Wirehair_Success) return 1;
    }
    data_ptr += m_size_data_submsg;
    encode_ptr += m_size_parity_submsg + m_size_data_submsg;
  }
  return 0;
}

int WirehairBenchmark::decode() noexcept {
  uint8_t* data_ptr = m_data_buffer;
  uint8_t* encode_ptr = m_encode_buffer;
  uint8_t* bitmap_ptr = m_block_bitmap;

  for (unsigned i = 0; i < m_num_chunks; ++i) {
    WirehairResult decode_result = Wirehair_NeedMore;
    m_decoder = wirehair_decoder_create(m_decoder, m_size_blk * m_data_blks_per_chunk, m_size_blk);
    if (!m_decoder) return 1;

    for (unsigned j = 0; j < m_num_total_blocks; ++j) {
      if (!bitmap_ptr[j]) continue; // lost this block
      decode_result = wirehair_decode(m_decoder, j, encode_ptr + j * m_size_blk, m_size_blk);
      if (decode_result == Wirehair_Success) break;
    }
    WirehairResult recover_result = wirehair_recover(m_decoder, data_ptr, m_size_data_submsg);
    if (recover_result != Wirehair_Success) return 1;

    data_ptr += m_size_data_submsg;
    encode_ptr += m_size_parity_submsg + m_size_data_submsg;
    bitmap_ptr += m_blks_per_chunk;
  }
  return 0;
}

void WirehairBenchmark::simulate_data_loss() noexcept {
  // TODO
  return;
}