#include "xorec_gpu_cmp_bm.hpp"
#include "xorec_gpu_cmp.cuh"
#include "utils.hpp"

XorecBenchmarkGpuCmp::XorecBenchmarkGpuCmp(const BenchmarkConfig& config) noexcept : AbstractBenchmark(config) {
  // Overwrite default initializations
  m_data_buf = make_unique_cuda<uint8_t>(m_block_size * m_num_data_blocks);
  m_parity_buf = make_unique_cuda<uint8_t>(m_block_size * m_num_parity_blocks);
  xorec_gpu_init(FIXED_GPU_BLOCKS, FIXED_GPU_THREADS_PER_BLOCK, m_num_data_blocks);
}

void XorecBenchmarkGpuCmp::setup() noexcept {
  std::fill_n(m_block_bitmap.get(), m_num_tot_blocks, 1);
  m_write_data_buffer();
}

void XorecBenchmarkGpuCmp::m_write_data_buffer() noexcept {
  std::unique_ptr<uint8_t[]> temp_data_buffer = std::make_unique<uint8_t[]>(m_block_size * m_num_data_blocks);
  for (unsigned i = 0; i < m_num_data_blocks; ++i) {
    if (write_validation_pattern(i, &temp_data_buffer[i*m_block_size], m_block_size)) {
      throw_error("Xorec (Gpu Computation): Failed to write validation pattern");
    }
  }
  cudaMemcpy(m_data_buf.get(), temp_data_buffer.get(), m_block_size * m_num_data_blocks, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
}

int XorecBenchmarkGpuCmp::encode() noexcept {
  XorecResult res = xorec_gpu_encode(m_data_buf.get(), m_parity_buf.get(), m_block_size, m_num_data_blocks, m_num_parity_blocks);
  cudaDeviceSynchronize();
  return (res == XorecResult::Success) ? 0 : -1;
}

int XorecBenchmarkGpuCmp::decode() noexcept {
  XorecResult res = xorec_gpu_decode(m_data_buf.get(), m_parity_buf.get(), m_block_size, m_num_data_blocks, m_num_parity_blocks, m_block_bitmap.get());
  cudaDeviceSynchronize();
  return (res == XorecResult::Success) ? 0 : -1;
  return 0;
}

void XorecBenchmarkGpuCmp::simulate_data_loss() noexcept {
  select_lost_block_idxs(m_num_data_blocks, m_num_parity_blocks, m_num_lost_blocks, m_block_bitmap.get());
  unsigned i;
  for (i = 0; i < m_num_data_blocks; ++i) {
    if (!m_block_bitmap[i]) {
      cudaError_t err = cudaMemsetAsync(&m_data_buf[i*m_block_size], 0, m_block_size);
      if (err != cudaSuccess) throw_error("Xorec (Gpu Computation): Failed to memset in simulate_data_loss.");
    }
  }

  for (; i < m_num_tot_blocks; ++i) {
    auto idx = i - m_num_data_blocks;
    if (!m_block_bitmap[i]) {
      cudaError_t err = cudaMemsetAsync(&m_parity_buf[idx*m_block_size], 0, m_block_size);
      if (err != cudaSuccess) throw_error("Xorec (Gpu Computation): Failed to memset in simulate_data_loss.");
    }
  }
  cudaDeviceSynchronize();
}

bool XorecBenchmarkGpuCmp::check_for_corruption() const noexcept {
  std::unique_ptr<uint8_t[]> temp_data_buffer = std::make_unique<uint8_t[]>(m_block_size * m_num_data_blocks);
  cudaMemcpy(temp_data_buffer.get(), m_data_buf.get(), m_block_size * m_num_data_blocks, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  for (unsigned i = 0; i < m_num_data_blocks; ++i) {
    if (!validate_block(&temp_data_buffer[i*m_block_size], m_block_size)) return false;
  }
  return true;
}
