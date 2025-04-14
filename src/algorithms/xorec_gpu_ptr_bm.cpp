#include "xorec.hpp"
#include "xorec_gpu_ptr_bm.hpp"
#include "utils.hpp"
#include <cuda_runtime.h>

XorecBenchmarkGpuPtr::XorecBenchmarkGpuPtr(const BenchmarkConfig& config) noexcept
  : AbstractBenchmark(config),
    m_version(config.xorec_version),
    m_gpu_data_buf(make_unique_cuda<uint8_t>(m_block_size * m_num_data_blocks))
{

  // Overwrite default initialization
  m_data_buf = make_unique_cuda_host<uint8_t>(m_block_size * m_num_data_blocks);
  xorec_init(m_num_data_blocks);
}

void XorecBenchmarkGpuPtr::setup() noexcept {
  std::fill_n(m_block_bitmap.get(), m_num_tot_blocks, 1);
  m_write_data_buffer();
  cudaError_t err = cudaMemcpy(m_gpu_data_buf.get(), m_data_buf.get(), m_block_size * m_num_data_blocks, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) throw_error("Xorec: Failed to copy data buffer to GPU.");
  std::fill_n(m_data_buf.get(), m_block_size * m_num_data_blocks, 0);
  cudaDeviceSynchronize();
}

int XorecBenchmarkGpuPtr::encode() noexcept {
  XorecResult res = xorec_gpu_encode(m_gpu_data_buf.get(), m_data_buf.get(), m_parity_buf.get(), m_block_size, m_num_data_blocks, m_num_parity_blocks, m_version);
  cudaDeviceSynchronize();
  return (res == XorecResult::Success) ? 0 : -1;
}

int XorecBenchmarkGpuPtr::decode() noexcept {
  XorecResult res = xorec_gpu_decode(m_gpu_data_buf.get(), m_data_buf.get(), m_parity_buf.get(), m_block_size, m_num_data_blocks, m_num_parity_blocks, m_block_bitmap.get(), m_version);
  cudaDeviceSynchronize();
  return (res == XorecResult::Success) ? 0 : -1;
}

void XorecBenchmarkGpuPtr::simulate_data_loss() noexcept {
  select_lost_block_idxs(m_num_data_blocks, m_num_parity_blocks, m_num_lost_blocks, m_block_bitmap.get());
  unsigned i;
  for (i = 0; i < m_num_data_blocks; ++i) {
    if (!m_block_bitmap[i]) {
      cudaError_t err = cudaMemsetAsync(&m_gpu_data_buf[i*m_block_size], 0, m_block_size);
      if (err != cudaSuccess) throw_error("Xorec (Gpu Computation): Failed to memset in simulate_data_loss.");
    }
  }
  for (; i < m_num_tot_blocks; ++i) {
    auto idx = i - m_num_data_blocks;
    if (!m_block_bitmap[i]) memset(&m_parity_buf[idx*m_block_size], 0, m_block_size);
  }
  cudaDeviceSynchronize();
}

bool XorecBenchmarkGpuPtr::check_for_corruption() const noexcept {
  cudaMemcpy(m_data_buf.get(), m_gpu_data_buf.get(), m_block_size * m_num_data_blocks, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (unsigned i = 0; i < m_num_data_blocks; ++i) {
    if (!validate_block(&m_data_buf[i*m_block_size], m_block_size)) return false;
  }
  return true;
}