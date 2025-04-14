#include "xorec.hpp"
#include "xorec_unified_ptr_bm.hpp"
#include "utils.hpp"
#include <cuda_runtime.h>

XorecBenchmarkUnifiedPtr::XorecBenchmarkUnifiedPtr(const BenchmarkConfig& config) noexcept
  : AbstractBenchmark(config),
    m_version(config.xorec_params.version)
{
  // Overwrite default initialization
  m_data_buf = make_unique_cuda_managed<uint8_t>(m_block_size * m_num_data_blocks);
  xorec_init(m_num_data_blocks);

  // Initialize data buffer with CRC blocks
  m_write_data_buffer();
  cudaDeviceSynchronize();
}

int XorecBenchmarkUnifiedPtr::encode() noexcept {
  XorecResult res = xorec_unified_encode(m_data_buf.get(), m_parity_buf.get(), m_block_size, m_num_data_blocks, m_num_parity_blocks, m_version);
  return (res == XorecResult::Success) ? 0 : -1;
}

int XorecBenchmarkUnifiedPtr::decode() noexcept {
  XorecResult res = xorec_unified_decode(m_data_buf.get(), m_parity_buf.get(), m_block_size, m_num_data_blocks, m_num_parity_blocks, m_block_bitmap.get(), m_version);
  return (res == XorecResult::Success) ? 0 : -1;
}

void XorecBenchmarkUnifiedPtr::touch_unified_memory() noexcept {
  touch_memory(m_data_buf.get(), m_block_size * m_num_data_blocks);
}
