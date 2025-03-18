#include "xorec.hpp"
#include "xorec_gpu_ptr_benchmark.hpp"
#include "utils.hpp"
#include <cuda_runtime.h>

XorecBenchmarkGpuPtr::XorecBenchmarkGpuPtr(const BenchmarkConfig& config) noexcept : ECBenchmark(config) {
  xorec_init();
  m_num_total_blocks = m_num_original_blocks + m_num_recovery_blocks;

  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&m_gpu_data_buffer), m_block_size * m_num_original_blocks);
  if (err != cudaSuccess) throw_error("Xorec: Failed to allocate GPU data buffer.");

  err = cudaMallocHost(reinterpret_cast<void**>(&m_cpu_data_buffer), m_block_size * m_num_original_blocks);
  if (err != cudaSuccess) throw_error("Xorec: Failed to allocate CPU data buffer.");

  m_parity_buffer = std::make_unique<uint8_t[]>(m_block_size * m_num_recovery_blocks);
  if (!m_parity_buffer) throw_error("Xorec: Failed to allocate parity buffer.");

  m_block_bitmap = std::make_unique<uint8_t[]>(XOREC_MAX_TOTAL_BLOCKS);
  m_version = config.xorec_params.version;

  for (unsigned i = 0; i < m_num_original_blocks; ++i) {
    int write_res = write_validation_pattern(i, &m_cpu_data_buffer[i * m_block_size], m_block_size);
    if (write_res) throw_error("Xorec: Failed to write random checking packet.");
  }

  err = cudaMemcpy(m_gpu_data_buffer, m_cpu_data_buffer, m_block_size * m_num_original_blocks, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) throw_error("Xorec: Failed to copy data buffer to GPU.");

  memset(m_cpu_data_buffer, 0, m_block_size * m_num_original_blocks);
  cudaDeviceSynchronize();
}

XorecBenchmarkGpuPtr::~XorecBenchmarkGpuPtr() noexcept {
  if (m_gpu_data_buffer) cudaFree(m_gpu_data_buffer);
  if (m_cpu_data_buffer) cudaFreeHost(m_cpu_data_buffer);
  cudaDeviceSynchronize();
}

int XorecBenchmarkGpuPtr::encode() noexcept {
  xorec_gpu_prefetch_encode(m_gpu_data_buffer, m_cpu_data_buffer, m_parity_buffer.get(), m_block_size, m_num_original_blocks, m_num_recovery_blocks, 0, m_version);
  return 0;
}

int XorecBenchmarkGpuPtr::decode() noexcept {
  xorec_gpu_prefetch_decode(m_gpu_data_buffer, m_cpu_data_buffer, m_parity_buffer.get(), m_block_size, m_num_original_blocks, m_num_recovery_blocks, m_block_bitmap.get(), 0, m_version);
  cudaDeviceSynchronize();
  return 0;
}

void XorecBenchmarkGpuPtr::simulate_data_loss() noexcept {
  unsigned loss_idx = 0;
  for (unsigned i = 0; i < m_num_total_blocks; ++i) {
    if (loss_idx < m_num_lost_blocks && m_lost_block_idxs[loss_idx] == i) {
      if (i < m_num_original_blocks) {
        cudaError_t err = cudaMemset(&m_gpu_data_buffer[i * m_block_size], 0, m_block_size);
        if (err != cudaSuccess) throw_error("Xorec: Failed to memset data buffer.");
        m_block_bitmap[i] = 0;
      } else {
        memset(&m_parity_buffer[(i - m_num_original_blocks) * m_block_size], 0, m_block_size);
        m_block_bitmap[i - m_num_original_blocks + XOREC_MAX_DATA_BLOCKS] = 0;
      }
      ++loss_idx;
      continue;
    }
    if (i < m_num_original_blocks) {
      m_block_bitmap[i] = 1;
    } else {
      m_block_bitmap[i - m_num_original_blocks + XOREC_MAX_DATA_BLOCKS] = 1;
    }
  }
  cudaDeviceSynchronize();
}

bool XorecBenchmarkGpuPtr::check_for_corruption() const noexcept {
  cudaMemcpy(m_cpu_data_buffer, m_gpu_data_buffer, m_block_size * m_num_original_blocks, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (unsigned i = 0; i < m_num_original_blocks; ++i) {
    if (!validate_block(&m_cpu_data_buffer[i * m_block_size], m_block_size)) return false;
  }
  return true;
}