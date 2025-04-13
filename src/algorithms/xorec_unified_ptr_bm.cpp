#include "xorec.hpp"
#include "xorec_unified_ptr_bm.hpp"
#include "utils.hpp"
#include <cuda_runtime.h>

XorecBenchmarkUnifiedPtr::XorecBenchmarkUnifiedPtr(const BenchmarkConfig& config) noexcept : AbstractBenchmark(config) {
  xorec_init(m_num_data_blocks);
  m_version = config.xorec_params.version;

  cudaError_t err = cudaMallocManaged(reinterpret_cast<void**>(&m_data_buf), m_block_size * m_num_data_blocks, cudaMemAttachHost); 
  if (err != cudaSuccess) throw_error("Xorec: Failed to allocate data buffer.");

  m_parity_buf = reinterpret_cast<uint8_t*>(_mm_malloc(m_block_size * m_num_parity_blocks, 64));
  m_block_bitmap = reinterpret_cast<uint8_t*>(_mm_malloc(m_num_tot_blocks, ALIGNMENT));
  if (!m_block_bitmap || !m_block_bitmap) throw_error("XorecBenchmark: Failed to allocate memory.");

  memset(m_block_bitmap, 1, m_num_tot_blocks);

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < m_num_data_blocks; ++i) {
    if (write_validation_pattern(i, m_data_buf+i*m_block_size, m_block_size)) throw_error("XorecBenchmark: Failed to write validation pattern");
  }
  cudaDeviceSynchronize();
}

XorecBenchmarkUnifiedPtr::~XorecBenchmarkUnifiedPtr() noexcept {
  cudaFree(m_data_buf);
  _mm_free(m_parity_buf);
  _mm_free(m_block_bitmap);
  cudaDeviceSynchronize();
}

int XorecBenchmarkUnifiedPtr::encode() noexcept {
  XorecResult res = xorec_unified_encode(m_data_buf, m_parity_buf, m_block_size, m_num_data_blocks, m_num_parity_blocks, m_version);
  return (res == XorecResult::Success) ? 0 : -1;
}

int XorecBenchmarkUnifiedPtr::decode() noexcept {
  XorecResult res = xorec_unified_decode(m_data_buf, m_parity_buf, m_block_size, m_num_data_blocks, m_num_parity_blocks, m_block_bitmap, m_version);
  return (res == XorecResult::Success) ? 0 : -1;
}

void XorecBenchmarkUnifiedPtr::simulate_data_loss() noexcept {
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
    if (!m_block_bitmap[i]) memset(m_parity_buf + idx * m_block_size, 0, m_block_size);
  }
  cudaDeviceSynchronize();
}

void XorecBenchmarkUnifiedPtr::touch_unified_memory() noexcept {
  touch_memory(m_data_buf, m_block_size * m_num_data_blocks);
}
