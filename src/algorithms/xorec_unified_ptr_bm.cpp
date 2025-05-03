#include "xorec.hpp"
#include "xorec_unified_ptr_bm.hpp"
#include "utils.hpp"
#include <cuda_runtime.h>

XorecBenchmarkUnifiedPtr::XorecBenchmarkUnifiedPtr(const BenchmarkConfig& config) noexcept
  : AbstractBenchmark(config),
    m_version(config.xorec_version)
{
  // Overwrite default initialization
  m_data_buf = make_unique_cuda_managed<uint8_t>(m_chunks * m_chunk_data_size);
  xorec_init(m_chunk_data_blocks);
}

void XorecBenchmarkUnifiedPtr::setup() noexcept {
  // Initialize data buffer with CRC blocks
  std::fill_n(m_block_bitmap.get(), m_chunks * m_chunk_tot_blocks, 1);
  m_write_data_buffer();
  m_touch_unified_memory();
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) throw_error("setup: cudaDeviceSynchronize failed");
  omp_set_num_threads(m_threads);
}

int XorecBenchmarkUnifiedPtr::encode() noexcept {
  int return_code = 0;

  #pragma omp parallel for
  for (unsigned c = 0; c < m_chunks; ++c) {
    auto data_buf = m_data_buf.get() + c * m_chunk_data_size;
    auto parity_buf = m_parity_buf.get() + c * m_chunk_parity_size;

    if (xorec_unified_encode(data_buf, parity_buf, m_block_size, m_chunk_data_blocks, m_chunk_parity_blocks, m_version) != XorecResult::Success) {
      #pragma omp atomic write
      return_code = 1;
    }
  }
  return return_code;
}

int XorecBenchmarkUnifiedPtr::decode() noexcept {
  int return_code = 0;

  #pragma omp parallel for
  for (unsigned c = 0; c < m_chunks; ++c) {
    auto bitmap = m_block_bitmap.get() + c * m_chunk_tot_blocks;
    auto data_buf = m_data_buf.get() + c * m_chunk_data_size;
    auto parity_buf = m_parity_buf.get() + c * m_chunk_parity_size;

    if (xorec_unified_decode(data_buf, parity_buf, m_block_size, m_chunk_data_blocks, m_chunk_parity_blocks, bitmap, m_version) != XorecResult::Success) {
      #pragma omp atomic write
      return_code = 1;
    }
  }
  return return_code;
}

void XorecBenchmarkUnifiedPtr::simulate_data_loss() noexcept {
  for (unsigned c = 0; c < m_chunks; ++c) {
    auto bitmap = m_block_bitmap.get() + c * m_chunk_tot_blocks;
    auto data_buf = m_data_buf.get() + c * m_chunk_data_size;
    auto parity_buf = m_parity_buf.get() + c * m_chunk_parity_size;

    select_lost_blocks(m_chunk_data_blocks, m_chunk_parity_blocks, m_chunk_lost_blocks, bitmap);

    unsigned i;
    for (i = 0; i < m_chunk_data_blocks; ++i) {
      if (!bitmap[i]) memset(&data_buf[i*m_block_size], 0, m_block_size);
    }

    for (; i < m_chunk_tot_blocks; ++i) {
      auto idx = i - m_chunk_data_blocks;
      if (!bitmap[i]) memset(&parity_buf[idx*m_block_size], 0, m_block_size);
    }
  }

  m_touch_unified_memory();
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) throw_error("simulate_data_loss: cudaDeviceSynchronize failed");
}

void XorecBenchmarkUnifiedPtr::m_touch_unified_memory() noexcept {
  int device_id = 0;
  cudaError_t err = cudaMemPrefetchAsync(m_data_buf.get(), m_chunks * m_chunk_data_size, device_id);
  if (err != cudaSuccess) throw_error("touch_memory: cudaMemPrefetchAsync failed");
}
