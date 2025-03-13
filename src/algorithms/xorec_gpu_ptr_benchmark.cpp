
#include "xorec.hpp"
#include "xorec_gpu_ptr_benchmark.hpp"
#include "cuda_utils.cuh"
#include "utils.hpp"

XorecBenchmarkGPUPtr::XorecBenchmarkGPUPtr(const BenchmarkConfig& config) noexcept : ECBenchmark(config) {
  xorec_init();
  m_num_total_blocks = m_num_original_blocks + m_num_recovery_blocks;
  cudaError_t err = cudaMallocManaged(reinterpret_cast<void**>(&m_data_buffer), m_block_size * m_num_original_blocks, cudaMemAttachHost); 
  if (err != cudaSuccess) throw_error("Xorec: Failed to allocate data buffer.");

  m_parity_buffer = std::make_unique<uint8_t[]>(m_block_size * m_num_recovery_blocks);
  if (!m_parity_buffer) throw_error("Xorec: Failed to allocate parity buffer.");

  m_block_bitmap = std::make_unique<uint8_t[]>(XOREC_MAX_TOTAL_BLOCKS);

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < m_num_original_blocks; ++i) {
    int write_res = write_validation_pattern(i, &m_data_buffer[i * m_block_size], m_block_size);
    if (write_res) throw_error("Xorec: Failed to write random checking packet.");
  }
}

XorecBenchmarkGPUPtr::~XorecBenchmarkGPUPtr() noexcept {
  if (m_data_buffer) cudaFree(m_data_buffer);
}

void XorecBenchmarkGPUPtr::simulate_data_loss() noexcept {
  uint32_t loss_idx = 0;
  for (unsigned i = 0; i < m_num_total_blocks; ++i) {
    if (loss_idx < m_num_lost_blocks && m_lost_block_idxs[loss_idx] == i) {
      if (i < m_num_original_blocks) {
        cudaError_t err = cudaMemset(&m_data_buffer[i * m_block_size], 0, m_block_size);
        if (err != cudaSuccess) throw_error("Xorec: Failed to memset data buffer.");
        m_block_bitmap[i] = 0;
      } else {
        memset(&m_parity_buffer[(i - m_num_original_blocks) * m_block_size], 0, m_block_size);
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
}

bool XorecBenchmarkGPUPtr::check_for_corruption() const noexcept {
  for (unsigned i = 0; i < m_num_original_blocks; ++i) {
    if (!validate_block(&m_data_buffer[i * m_block_size], m_block_size)) return false;
  }
  return true;
}

void XorecBenchmarkGPUPtr::touch_gpu_memory() noexcept {
  touch_memory(m_data_buffer, m_block_size * m_num_original_blocks);
}




XorecBenchmarkScalarGPUPtr::XorecBenchmarkScalarGPUPtr(const BenchmarkConfig& config) noexcept : XorecBenchmarkGPUPtr(config) {}
int XorecBenchmarkScalarGPUPtr::encode() noexcept {
  xorec_encode(m_data_buffer, m_parity_buffer.get(), m_block_size, m_num_original_blocks, m_num_recovery_blocks, XorecVersion::Scalar);
  return 0;
}
int XorecBenchmarkScalarGPUPtr::decode() noexcept {
  xorec_decode(m_data_buffer, m_parity_buffer.get(), m_block_size, m_num_original_blocks, m_num_recovery_blocks, m_block_bitmap.get(), XorecVersion::Scalar);
  return 0;
}

XorecBenchmarkAVXGPUPtr::XorecBenchmarkAVXGPUPtr(const BenchmarkConfig& config) noexcept : XorecBenchmarkGPUPtr(config) {}
int XorecBenchmarkAVXGPUPtr::encode() noexcept {
  xorec_encode(m_data_buffer, m_parity_buffer.get(), m_block_size, m_num_original_blocks, m_num_recovery_blocks, XorecVersion::AVX);
  return 0;
}
int XorecBenchmarkAVXGPUPtr::decode() noexcept {
  xorec_decode(m_data_buffer, m_parity_buffer.get(), m_block_size, m_num_original_blocks, m_num_recovery_blocks, m_block_bitmap.get(), XorecVersion::AVX);
  return 0;
}

XorecBenchmarkAVX2GPUPtr::XorecBenchmarkAVX2GPUPtr(const BenchmarkConfig& config) noexcept : XorecBenchmarkGPUPtr(config) {}
int XorecBenchmarkAVX2GPUPtr::encode() noexcept {
  xorec_encode(m_data_buffer, m_parity_buffer.get(), m_block_size, m_num_original_blocks, m_num_recovery_blocks, XorecVersion::AVX2);
  return 0;
}
int XorecBenchmarkAVX2GPUPtr::decode() noexcept {
  xorec_decode(m_data_buffer, m_parity_buffer.get(), m_block_size, m_num_original_blocks, m_num_recovery_blocks, m_block_bitmap.get(), XorecVersion::AVX2);
  return 0;
}