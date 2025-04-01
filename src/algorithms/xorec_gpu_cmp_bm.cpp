#include "xorec_gpu_cmp_bm.hpp"
#include "xorec_gpu_cmp.cuh"
#include "utils.hpp"

XorecBenchmarkGpuCmp::XorecBenchmarkGpuCmp(const BenchmarkConfig& config) noexcept : ECBenchmark(config) {
  xorec_gpu_init(FIXED_GPU_BLOCKS, FIXED_GPU_THREADS_PER_BLOCK, m_num_data_blocks);

  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&m_data_buf), m_block_size * m_num_data_blocks);
  if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err)<< '\n';
    throw_error("Xorec (Gpu Computation): Failed to allocate data buffer.");
  }

  err = cudaMalloc(reinterpret_cast<void**>(&m_parity_buf), m_block_size * m_num_parity_blocks);
  if (err != cudaSuccess) throw_error("Xorec (Gpu Computation): Failed to allocate parity buffer.");

  m_block_bitmap = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_tot_blocks, ALIGNMENT));
  if (!m_block_bitmap) throw_error("Xorec (Gpu Computation): Failed to allocate block bitmap.");
  memset(m_block_bitmap, 1, m_num_tot_blocks);
  
  std::unique_ptr<uint8_t[]> temp_data_buffer = std::make_unique<uint8_t[]>(m_block_size * m_num_data_blocks);

  for (unsigned i = 0; i < m_num_data_blocks; ++i) {
    if (write_validation_pattern(i, &temp_data_buffer[i*m_block_size], m_block_size)) throw_error("Xorec (Gpu Computation): Failed to write validation pattern");
  }

  cudaMemcpy(m_data_buf, temp_data_buffer.get(), m_block_size * m_num_data_blocks, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
}

XorecBenchmarkGpuCmp::~XorecBenchmarkGpuCmp() noexcept {
  cudaFree(m_data_buf);
  cudaFree(m_parity_buf);
  _mm_free(m_block_bitmap);
}

int XorecBenchmarkGpuCmp::encode() noexcept {
  XorecResult res = xorec_gpu_encode(m_data_buf, m_parity_buf, m_block_size, m_num_data_blocks, m_num_parity_blocks);
  cudaDeviceSynchronize();
  return (res == XorecResult::Success) ? 0 : -1;
}

int XorecBenchmarkGpuCmp::decode() noexcept {
  XorecResult res = xorec_gpu_decode(m_data_buf, m_parity_buf, m_block_size, m_num_data_blocks, m_num_parity_blocks, m_block_bitmap);
  cudaDeviceSynchronize();
  return (res == XorecResult::Success) ? 0 : -1;
  return 0;
}

void XorecBenchmarkGpuCmp::simulate_data_loss() noexcept {
  select_lost_block_idxs(m_num_data_blocks, m_num_parity_blocks, m_num_lost_blocks, m_block_bitmap);
  unsigned i;
  for (i = 0; i < m_num_data_blocks; ++i) {
    if (!m_block_bitmap[i]) {
      cudaError_t err = cudaMemsetAsync(m_data_buf + i * m_block_size, 0, m_block_size);
      if (err != cudaSuccess) throw_error("Xorec (Gpu Computation): Failed to memset in simulate_data_loss.");
    }
  }

  for (; i < m_num_tot_blocks; ++i) {
    auto idx = i - m_num_data_blocks;
    if (!m_block_bitmap[i]) {
      cudaError_t err = cudaMemsetAsync(m_parity_buf + idx * m_block_size, 0, m_block_size);
      if (err != cudaSuccess) throw_error("Xorec (Gpu Computation): Failed to memset in simulate_data_loss.");
    }
  }
  cudaDeviceSynchronize();
}

bool XorecBenchmarkGpuCmp::check_for_corruption() const noexcept {
  std::unique_ptr<uint8_t[]> temp_data_buffer = std::make_unique<uint8_t[]>(m_block_size * m_num_data_blocks);
  cudaMemcpy(temp_data_buffer.get(), m_data_buf, m_block_size * m_num_data_blocks, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  for (unsigned i = 0; i < m_num_data_blocks; ++i) {
    if (!validate_block(&temp_data_buffer[i * m_block_size], m_block_size)) return false;
  }
  return true;
}
