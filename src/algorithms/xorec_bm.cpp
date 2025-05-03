/**
 * @file xorec_bm.cpp
 * @brief Benchmark implementation for the XOR-EC implementation.
 * 
 * Documentation can be found in xorec.h and abstract_bm.h
 */


#include "xorec_bm.hpp"
#include "xorec.hpp"
#include "utils.hpp"
 
 
XorecBenchmark::XorecBenchmark(const BenchmarkConfig& config) noexcept
  : AbstractBenchmark(config),
    m_version(config.xorec_version)
{
  xorec_init(m_chunk_data_blocks);
}

void XorecBenchmark::setup() noexcept {
  std::fill_n(m_block_bitmap.get(), m_chunks * m_chunk_tot_blocks, 1);
  m_write_data_buffer();
  omp_set_num_threads(m_threads);
}

int XorecBenchmark::encode() noexcept {
  int return_code = 0;

  #pragma omp parallel for
  for (unsigned c = 0; c < m_chunks; ++c) {
    auto data_buf = m_data_buf.get() + c * m_chunk_data_size;
    auto parity_buf = m_parity_buf.get() + c * m_chunk_parity_size;

    if (xorec_encode(data_buf, parity_buf, m_block_size, m_chunk_data_blocks, m_chunk_parity_blocks, m_version) != XorecResult::Success) {
      #pragma omp atomic write
      return_code = 1;
    }
  }
  return return_code;
}

int XorecBenchmark::decode() noexcept {
  int return_code = 0;

  #pragma omp parallel for
  for (unsigned c = 0; c < m_chunks; ++c) {
    auto bitmap = m_block_bitmap.get() + c * m_chunk_tot_blocks;
    auto data_buf = m_data_buf.get() + c * m_chunk_data_size;
    auto parity_buf = m_parity_buf.get() + c * m_chunk_parity_size;

    if (xorec_decode(data_buf, parity_buf, m_block_size, m_chunk_data_blocks, m_chunk_parity_blocks, bitmap, m_version) != XorecResult::Success) {
      #pragma omp atomic write
      return_code = 1;
    }
  }
  return return_code;
}
