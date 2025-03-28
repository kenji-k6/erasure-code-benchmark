#include "xorec_gpu_cmp_bm.hpp"
#include "xorec_gpu_cmp.cuh"
#include "utils.hpp"

XorecBenchmarkGpuCmp::XorecBenchmarkGpuCmp(const BenchmarkConfig& config) noexcept : ECBenchmark(config) {
  if (!config.is_gpu_config) throw_error("Xorec (Gpu Computation): Invalid configuration for GPU benchmark.");

  xorec_gpu_init(config.num_gpu_blocks, config.threads_per_gpu_block, m_data_blks_per_chunk, m_parity_blks_per_chunk);

  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&m_data_buffer), m_num_chunks * m_size_data_submsg);
  if (err != cudaSuccess) throw_error("Xorec (Gpu Computation): Failed to allocate data buffer.");

  err = cudaMalloc(reinterpret_cast<void**>(&m_parity_buffer), m_num_chunks * m_size_parity_submsg);
  if (err != cudaSuccess) throw_error("Xorec (Gpu Computation): Failed to allocate parity buffer.");

  m_block_bitmap = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_chunks * m_blks_per_chunk, ALIGNMENT));
  if (!m_block_bitmap) throw_error("Xorec (Gpu Computation): Failed to allocate block bitmap.");
  memset(m_block_bitmap, 1, m_num_chunks * m_blks_per_chunk);


  std::unique_ptr<uint8_t[]> temp_data_buffer = std::make_unique<uint8_t[]>(m_num_chunks * m_size_data_submsg);

  for (unsigned i = 0; i < m_size_msg/m_size_blk; ++i) {
    if (write_validation_pattern(i, &temp_data_buffer[i*m_size_blk], m_size_blk)) throw_error("Xorec (Gpu Computation): Failed to write validation pattern");
  }
  cudaMemcpy(m_data_buffer, temp_data_buffer.get(), m_num_chunks * m_size_data_submsg, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
}

XorecBenchmarkGpuCmp::~XorecBenchmarkGpuCmp() noexcept {
  cudaFree(m_data_buffer);
  cudaFree(m_parity_buffer);
  _mm_free(m_block_bitmap);
}

int XorecBenchmarkGpuCmp::encode() noexcept {
  xorec_gpu_encode_full_message(m_data_buffer, m_parity_buffer, m_num_chunks, m_size_blk, m_data_blks_per_chunk, m_parity_blks_per_chunk);
  cudaDeviceSynchronize();
  return 0;
}

int XorecBenchmarkGpuCmp::decode() noexcept {
  uint8_t* data_ptr = m_data_buffer;
  uint8_t* parity_ptr = m_parity_buffer;
  uint8_t* bitmap_ptr = m_block_bitmap;

  for (unsigned i = 0; i < m_num_chunks; ++i) {
    xorec_gpu_decode(data_ptr, parity_ptr, m_size_blk, m_data_blks_per_chunk, m_parity_blks_per_chunk, bitmap_ptr);
    data_ptr += m_size_data_submsg;
    parity_ptr += m_size_parity_submsg;
    bitmap_ptr += m_blks_per_chunk;
  }
  cudaDeviceSynchronize();
  return 0;
}

void XorecBenchmarkGpuCmp::simulate_data_loss() noexcept {
  // TODO
  return;
}

bool XorecBenchmarkGpuCmp::check_for_corruption() const noexcept {
  std::unique_ptr<uint8_t[]> temp_data_buffer = std::make_unique<uint8_t[]>(m_num_chunks * m_size_data_submsg);
  cudaMemcpy(temp_data_buffer.get(), m_data_buffer, m_num_chunks*m_size_data_submsg, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  for (unsigned i = 0; i < m_size_msg/m_size_blk; ++i) {
    if (!validate_block(&temp_data_buffer[i * m_size_blk], m_size_blk)) return false;
  }
  return true;
}