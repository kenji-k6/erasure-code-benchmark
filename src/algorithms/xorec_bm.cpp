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
  xorec_init(m_num_data_blocks);
}

void XorecBenchmark::setup() noexcept {
  std::fill_n(m_block_bitmap.get(), m_num_tot_blocks, 1);
  m_write_data_buffer();
}

int XorecBenchmark::encode() noexcept {
  XorecResult res = xorec_encode(m_data_buf.get(), m_parity_buf.get(), m_block_size, m_num_data_blocks, m_num_parity_blocks, m_version);
  return (res == XorecResult::Success) ? 0 : -1;
}

int XorecBenchmark::decode() noexcept {
  XorecResult res = xorec_decode(m_data_buf.get(), m_parity_buf.get(), m_block_size, m_num_data_blocks, m_num_parity_blocks, m_block_bitmap.get(), m_version);
  return (res == XorecResult::Success) ? 0 : -1;
}
