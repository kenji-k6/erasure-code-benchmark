#include "xorec_gpu_cmp_bm.hpp"
#include "xorec_gpu_cmp.cuh"
#include "xorec.hpp"
#include "utils.hpp"

XorecBenchmarkGpuCmp::XorecBenchmarkGpuCmp(const BenchmarkConfig& config) noexcept
  : AbstractBenchmark(config),
    m_num_gpu_blocks(config.num_gpu_blocks),
    m_threads_per_gpu_block(config.threads_per_gpu_block)
{
  // Overwrite default initializations
  m_data_buf = make_unique_cuda<uint8_t>(m_chunks * m_chunk_data_size);
  m_parity_buf = make_unique_cuda<uint8_t>(m_chunks * m_chunk_parity_size);
  xorec_gpu_init(m_chunk_data_blocks);
  xorec_init(m_chunk_data_blocks);
}

void XorecBenchmarkGpuCmp::setup() noexcept {
  std::fill_n(m_block_bitmap.get(), m_chunks * m_chunk_tot_blocks, 1);
  m_write_data_buffer();
}

void XorecBenchmarkGpuCmp::m_write_data_buffer() noexcept {
  std::unique_ptr<uint8_t[]> temp_data_buffer = std::make_unique<uint8_t[]>(m_chunks * m_chunk_data_size);

  for (unsigned c = 0; c < m_chunks; ++c) {
    auto data_buf = temp_data_buffer.get() + c * m_chunk_data_size;
    for (unsigned i = 0; i < m_chunk_data_blocks; ++i) {
      if (write_validation_pattern(&data_buf[i*m_block_size], m_block_size)) {
        throw_error("Failed to write random checking packet.");
      }
    }
  }
  cudaMemcpy(m_data_buf.get(), temp_data_buffer.get(), m_chunks * m_chunk_data_size, cudaMemcpyHostToDevice);
}

int XorecBenchmarkGpuCmp::encode() noexcept {
  XorecResult res = xorec_gpu_encode(
    m_data_buf.get(),
    m_parity_buf.get(),
    m_chunks,
    m_block_size,
    m_chunk_data_blocks,
    m_chunk_parity_blocks,
    m_num_gpu_blocks,
    m_threads_per_gpu_block
  );
  cudaDeviceSynchronize();
  return (res == XorecResult::Success) ? 0 : -1;
}

int XorecBenchmarkGpuCmp::decode() noexcept {
  XorecResult res = xorec_gpu_decode(
    m_data_buf.get(),
    m_parity_buf.get(),
    m_chunks,
    m_block_size,
    m_chunk_data_blocks,
    m_chunk_parity_blocks,
    m_block_bitmap.get(),
    m_num_gpu_blocks,
    m_threads_per_gpu_block
  );
  cudaDeviceSynchronize();
  return (res == XorecResult::Success) ? 0 : -1;
}

void XorecBenchmarkGpuCmp::simulate_data_loss() noexcept {
  for (unsigned c = 0; c < m_chunks; ++c) {
    auto bitmap = m_block_bitmap.get() + c * m_chunk_tot_blocks;
    auto data_buf = m_data_buf.get() + c * m_chunk_data_size;
    auto parity_buf = m_parity_buf.get() + c * m_chunk_parity_size;

    select_lost_blocks(m_chunk_data_blocks, m_chunk_parity_blocks, m_chunk_lost_blocks, bitmap);

    unsigned i;
    for (i = 0; i < m_chunk_data_blocks; ++i) {
      if (!bitmap[i]) cudaMemset(&data_buf[i*m_block_size], 0, m_block_size);
    }

    for (; i < m_chunk_tot_blocks; ++i) {
      auto idx = i - m_chunk_data_blocks;
      if (!bitmap[i]) cudaMemset(&parity_buf[idx*m_block_size], 0, m_block_size);
    }
  }
}

bool XorecBenchmarkGpuCmp::check_for_corruption() const noexcept {
  auto temp_data_buf = make_unique_aligned<uint8_t>(m_chunks * m_chunk_data_size);
  auto temp_parity_buf = make_unique_aligned<uint8_t>(m_chunks * m_chunk_parity_size);
  cudaMemcpy(temp_data_buf.get(), m_data_buf.get(), m_chunks * m_chunk_data_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(temp_parity_buf.get(), m_parity_buf.get(), m_chunks * m_chunk_parity_size, cudaMemcpyDeviceToHost);

  for (unsigned c = 0; c < m_chunks; ++c) {
    auto data_buf = temp_data_buf.get() + c * m_chunk_data_size;
    for (unsigned i = 0; i < m_chunk_data_blocks; ++i) {
      if (!validate_block(&data_buf[i*m_block_size], m_block_size)) return false;
    }
  }
  return true;
}
