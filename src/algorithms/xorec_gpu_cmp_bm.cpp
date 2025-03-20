#include "xorec_gpu_cmp_bm.hpp"
#include "xorec_gpu_cmp.cuh"
#include "utils.hpp"

XorecBenchmarkGpuCmp::XorecBenchmarkGpuCmp(const BenchmarkConfig& config) noexcept : ECBenchmark(config) {
  xorec_gpu_init();

  m_num_total_blocks = m_num_original_blocks + m_num_recovery_blocks;
  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&m_data_buffer), m_block_size * m_num_original_blocks);
  if (err != cudaSuccess) throw_error("Xorec (Gpu Computation): Failed to allocate data buffer.");

  err = cudaMalloc(reinterpret_cast<void**>(&m_parity_buffer), m_block_size * m_num_recovery_blocks);
  if (err != cudaSuccess) throw_error("Xorec (Gpu Computation): Failed to allocate parity buffer.");

  m_block_bitmap = std::make_unique<uint8_t[]>(XOREC_MAX_TOTAL_BLOCKS);
  
  std::unique_ptr<uint8_t[]> temp_data_buffer = std::make_unique<uint8_t[]>(m_block_size * m_num_original_blocks);

  for (unsigned i = 0; i < m_num_original_blocks; ++i) {
    int write_res = write_validation_pattern(i, &temp_data_buffer[i * m_block_size], m_block_size);
    if (write_res) throw_error("Xorec (Gpu Computation): Failed to write random checking packet.");
  }

  cudaMemcpy(m_data_buffer, temp_data_buffer.get(), m_block_size * m_num_original_blocks, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
}

XorecBenchmarkGpuCmp::~XorecBenchmarkGpuCmp() noexcept {
  if (m_data_buffer) cudaFree(m_data_buffer);
  if (m_parity_buffer) cudaFree(m_parity_buffer);
}

int XorecBenchmarkGpuCmp::encode() noexcept {
  xorec_gpu_encode(m_data_buffer, m_parity_buffer, m_block_size, m_num_original_blocks, m_num_recovery_blocks);
  cudaDeviceSynchronize();
  return 0;
}

int XorecBenchmarkGpuCmp::decode() noexcept {
  xorec_gpu_decode(m_data_buffer, m_parity_buffer, m_block_size, m_num_original_blocks, m_num_recovery_blocks, m_block_bitmap.get());
  cudaDeviceSynchronize();
  return 0;
}

void XorecBenchmarkGpuCmp::simulate_data_loss() noexcept {
  unsigned loss_idx = 0;
  for (unsigned i = 0; i < m_num_total_blocks; ++i) {
    if (loss_idx < m_num_lost_blocks && m_lost_block_idxs[loss_idx] == i) {

      if (i < m_num_original_blocks) {
        cudaError_t err = cudaMemset(&m_data_buffer[i * m_block_size], 0, m_block_size);
        if (err != cudaSuccess) throw_error("Xorec (Gpu Computation): Failed to memset data buffer.");
        m_block_bitmap[i] = 0;
      } else {
        cudaError_t err = cudaMemset(&m_parity_buffer[(i - m_num_original_blocks) * m_block_size], 0, m_block_size);
        if (err != cudaSuccess) throw_error("Xorec (Gpu Computation): Failed to memset parity buffer.");
        m_block_bitmap[i-m_num_original_blocks + XOREC_MAX_DATA_BLOCKS] = 0;
      }

      ++loss_idx;
      continue;
    }
    if (i < m_num_original_blocks) {
      m_block_bitmap[i] = 1;
    } else {
      m_block_bitmap[i-m_num_original_blocks + XOREC_MAX_DATA_BLOCKS] = 1;
    }
  }
  cudaDeviceSynchronize();
}

bool XorecBenchmarkGpuCmp::check_for_corruption() const noexcept {
  std::unique_ptr<uint8_t[]> temp_data_buffer = std::make_unique<uint8_t[]>(m_block_size * m_num_original_blocks);
  cudaMemcpy(temp_data_buffer.get(), m_data_buffer, m_block_size * m_num_original_blocks, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  for (unsigned i = 0; i < m_num_original_blocks; ++i) {
    if (!validate_block(&temp_data_buffer[i * m_block_size], m_block_size)) return false;
  }
  return true;
}
