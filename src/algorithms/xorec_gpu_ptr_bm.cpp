#include "xorec.hpp"
#include "xorec_gpu_ptr_bm.hpp"
#include "utils.hpp"
#include <cuda_runtime.h>

XorecBenchmarkGpuPtr::XorecBenchmarkGpuPtr(const BenchmarkConfig& config) noexcept : ECBenchmark(config) {
  xorec_init(m_num_data_blocks);
  m_version = config.xorec_params.version;

  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&m_gpu_data_buf), m_block_size * m_num_data_blocks);
  if (err != cudaSuccess) throw_error("Xorec: Failed to allocate GPU data buffer.");

  err = cudaMallocHost(reinterpret_cast<void**>(&m_data_buf), m_block_size * m_num_data_blocks);
  if (err != cudaSuccess) throw_error("Xorec: Failed to allocate CPU data buffer.");

  m_parity_buf = reinterpret_cast<uint8_t*>(_mm_malloc(m_block_size * m_num_parity_blocks, 64));
  m_block_bitmap = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_tot_blocks, ALIGNMENT));
  if (!m_block_bitmap || !m_block_bitmap) throw_error("XorecBenchmark: Failed to allocate memory.");

  memset(m_block_bitmap, 1, m_num_tot_blocks);

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < m_num_data_blocks; ++i) {
    if (write_validation_pattern(i, m_data_buf+i*m_block_size, m_block_size)) throw_error("XorecBenchmark: Failed to write validation pattern");
  }

  err = cudaMemcpy(m_gpu_data_buf, m_data_buf, m_block_size * m_num_data_blocks, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) throw_error("Xorec: Failed to copy data buffer to GPU.");

  memset(m_data_buf, 0, m_block_size * m_num_data_blocks);
  cudaDeviceSynchronize();
}

XorecBenchmarkGpuPtr::~XorecBenchmarkGpuPtr() noexcept {
  cudaFree(m_gpu_data_buf);
  cudaFreeHost(m_data_buf);
  _mm_free(m_parity_buf);
  _mm_free(m_block_bitmap);
  cudaDeviceSynchronize();
}

int XorecBenchmarkGpuPtr::encode() noexcept {
  XorecResult res = xorec_gpu_encode(m_gpu_data_buf, m_data_buf, m_parity_buf, m_block_size, m_num_data_blocks, m_num_parity_blocks, m_version);
  cudaDeviceSynchronize();
  return (res == XorecResult::Success) ? 0 : -1;
}

int XorecBenchmarkGpuPtr::decode() noexcept {
  XorecResult res = xorec_gpu_decode(m_gpu_data_buf, m_data_buf, m_parity_buf, m_block_size, m_num_data_blocks, m_num_parity_blocks, m_block_bitmap, m_version);
  cudaDeviceSynchronize();
  return (res == XorecResult::Success) ? 0 : -1;
}

void XorecBenchmarkGpuPtr::simulate_data_loss() noexcept {
  // unsigned loss_idx = 0;
  // for (unsigned i = 0; i < m_num_tot_blocks; ++i) {
  //   if (loss_idx < m_num_lost_blocks && m_lost_block_idxs[loss_idx] == i) {
  //     if (i < m_num_data_blocks) {
  //       cudaError_t err = cudaMemset(&m_gpu_data_buffer[i * m_block_size], 0, m_block_size);
  //       if (err != cudaSuccess) throw_error("Xorec: Failed to memset data buffer.");
  //       m_block_bitmap[i] = 0;
  //     } else {
  //       memset(&m_parity_buffer[(i - m_num_data_blocks) * m_block_size], 0, m_block_size);
  //       m_block_bitmap[i - m_num_data_blocks + XOREC_MAX_DATA_BLOCKS] = 0;
  //     }
  //     ++loss_idx;
  //     continue;
  //   }
  //   if (i < m_num_data_blocks) {
  //     m_block_bitmap[i] = 1;
  //   } else {
  //     m_block_bitmap[i - m_num_data_blocks + XOREC_MAX_DATA_BLOCKS] = 1;
  //   }
  // }
  // cudaDeviceSynchronize();
}

bool XorecBenchmarkGpuPtr::check_for_corruption() const noexcept {
  cudaMemcpy(m_data_buf, m_gpu_data_buf, m_block_size * m_num_data_blocks, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (unsigned i = 0; i < m_num_data_blocks; ++i) {
    if (!validate_block(m_data_buf+i*m_block_size, m_block_size)) return false;
  }
  return true;
}