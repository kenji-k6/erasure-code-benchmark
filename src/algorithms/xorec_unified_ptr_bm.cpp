#include "xorec.hpp"
#include "xorec_unified_ptr_bm.hpp"
#include "utils.hpp"
#include <cuda_runtime.h>

XorecBenchmarkUnifiedPtr::XorecBenchmarkUnifiedPtr(const BenchmarkConfig& config) noexcept
  : AbstractBenchmark(config),
    m_version(config.xorec_version)
{
  // Overwrite default initialization
  m_data_buf = make_unique_cuda_managed<uint8_t>(m_block_size * m_num_data_blocks);
  xorec_init(m_num_data_blocks);
}

void XorecBenchmarkUnifiedPtr::setup() noexcept {
  // Initialize data buffer with CRC blocks
  std::fill_n(m_block_bitmap.get(), m_num_tot_blocks, 1);
  m_write_data_buffer();
  m_touch_unified_memory();
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) throw_error("setup: cudaDeviceSynchronize failed");
}

int XorecBenchmarkUnifiedPtr::encode() noexcept {
  XorecResult res = xorec_unified_encode(m_data_buf.get(), m_parity_buf.get(), m_block_size, m_num_data_blocks, m_num_parity_blocks, m_version);
  return (res == XorecResult::Success) ? 0 : -1;
}

int XorecBenchmarkUnifiedPtr::decode() noexcept {
  XorecResult res = xorec_unified_decode(m_data_buf.get(), m_parity_buf.get(), m_block_size, m_num_data_blocks, m_num_parity_blocks, m_block_bitmap.get(), m_version);
  return (res == XorecResult::Success) ? 0 : -1;
}

void XorecBenchmarkUnifiedPtr::simulate_data_loss() noexcept {
  select_lost_block_idxs(m_num_data_blocks, m_num_parity_blocks, m_num_lost_blocks, m_block_bitmap.get());

  unsigned i;
  for (i = 0; i < m_num_data_blocks; ++i) {
    if (!m_block_bitmap[i]) memset(&m_data_buf[i*m_block_size], 0, m_block_size);
  }

  for (; i < m_num_tot_blocks; ++i) {
    auto idx = i - m_num_data_blocks;
    if (!m_block_bitmap[i]) memset(&m_parity_buf[idx*m_block_size], 0, m_block_size);
  }

  m_touch_unified_memory();
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) throw_error("simulate_data_loss: cudaDeviceSynchronize failed");
}

void XorecBenchmarkUnifiedPtr::m_touch_unified_memory() noexcept {
  int device_id = 0;
  cudaError_t err = cudaMemPrefetchAsync(m_data_buf.get(), m_block_size * m_num_data_blocks, device_id);
  if (err != cudaSuccess) throw_error("touch_memory: cudaMemPrefetchAsync failed");
}
