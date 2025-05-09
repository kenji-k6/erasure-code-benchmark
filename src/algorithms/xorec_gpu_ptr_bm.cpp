#include "xorec.hpp"
#include "xorec_gpu_ptr_bm.hpp"
#include "utils.hpp"
#include <cuda_runtime.h>

XorecBenchmarkGpuPtr::XorecBenchmarkGpuPtr(const BenchmarkConfig& config) noexcept
  : AbstractBenchmark(config),
    m_version(config.xorec_version),
    m_gpu_data_buf(make_unique_cuda<uint8_t>(m_chunks * m_chunk_data_size))
{

  // Overwrite default initialization
  m_data_buf = make_unique_cuda_host<uint8_t>(m_chunks * m_chunk_data_size);
  xorec_init(m_chunk_data_blocks);
}

void XorecBenchmarkGpuPtr::setup() noexcept {
  std::fill_n(m_block_bitmap.get(), m_chunks * m_chunk_tot_blocks, 1);


  m_write_data_buffer();
  cudaError_t err = cudaMemcpy(m_gpu_data_buf.get(), m_data_buf.get(), m_chunks * m_chunk_data_size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) throw_error("Xorec: Failed to copy data buffer to GPU.");
  std::fill_n(m_data_buf.get(), m_chunks * m_chunk_data_size, 0);
  cudaDeviceSynchronize();
  omp_set_num_threads(m_threads);
}

int XorecBenchmarkGpuPtr::encode() noexcept {
  int return_code = 0;

  #pragma omp parallel for
  for (unsigned c = 0; c < m_chunks; ++c) {
    auto data_buf = m_data_buf.get() + c * m_chunk_data_size;
    auto gpu_data_buf = m_gpu_data_buf.get() + c * m_chunk_data_size;
    auto parity_buf = m_parity_buf.get() + c * m_chunk_parity_size;

    if (xorec_gpu_ptr_encode(gpu_data_buf, data_buf, parity_buf, m_block_size, m_chunk_data_blocks, m_chunk_parity_blocks, m_version) != XorecResult::Success) {
      #pragma omp atomic write
      return_code = 1;
    }
  }

  cudaDeviceSynchronize();
  return return_code;
}

int XorecBenchmarkGpuPtr::decode() noexcept {
  int return_code = 0;

  #pragma omp parallel for
  for (unsigned c = 0; c < m_chunks; ++c) {
    auto bitmap = m_block_bitmap.get() + c * m_chunk_tot_blocks;
    auto data_buf = m_data_buf.get() + c * m_chunk_data_size;
    auto gpu_data_buf = m_gpu_data_buf.get() + c * m_chunk_data_size;
    auto parity_buf = m_parity_buf.get() + c * m_chunk_parity_size;

    if (xorec_gpu_ptr_decode(gpu_data_buf, data_buf, parity_buf, m_block_size, m_chunk_data_blocks, m_chunk_parity_blocks, bitmap, m_version) != XorecResult::Success) {
      #pragma omp atomic write
      return_code = 1;
    }
  }

  cudaDeviceSynchronize();
  return return_code;
}

void XorecBenchmarkGpuPtr::simulate_data_loss() noexcept {

  for (unsigned c = 0; c < m_chunks; ++c) {
    auto bitmap = m_block_bitmap.get() + c * m_chunk_tot_blocks;
    auto gpu_data_buf = m_gpu_data_buf.get() + c * m_chunk_data_size;
    auto parity_buf = m_parity_buf.get() + c * m_chunk_parity_size;

    select_lost_blocks(m_chunk_data_blocks, m_chunk_parity_blocks, m_chunk_lost_blocks, bitmap);

    unsigned i;
    for (i = 0; i < m_chunk_data_blocks; ++i) {
      if (!bitmap[i]) cudaMemsetAsync(&gpu_data_buf[i*m_block_size], 0, m_block_size);
    }

    for (; i < m_chunk_tot_blocks; ++i) {
      auto idx = i - m_chunk_data_blocks;
      if (!bitmap[i]) memset(&parity_buf[idx*m_block_size], 0, m_block_size);
    }
  }

  cudaDeviceSynchronize();
}

bool XorecBenchmarkGpuPtr::check_for_corruption() const noexcept {

  cudaMemcpy(m_data_buf.get(), m_gpu_data_buf.get(), m_chunks * m_chunk_data_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  for (unsigned c = 0; c < m_chunks; ++c) {
    auto data_buf = m_data_buf.get() + c * m_chunk_data_size;

    for (unsigned i = 0; i < m_chunk_data_blocks; ++i) {
      if (!validate_block(&data_buf[i*m_block_size], m_block_size)) return false;
    }
  }
  return true;
}