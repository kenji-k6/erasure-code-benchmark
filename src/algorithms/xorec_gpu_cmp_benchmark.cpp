#include "xorec_gpu_cmp_benchmark.h"

XorecBenchmarkGPUCmp::XorecBenchmarkGPUCmp(const BenchmarkConfig& config) noexcept : ECBenchmark(config) {
  num_total_blocks_ = num_original_blocks_ + num_recovery_blocks_;
  
  cudaError_t err = cudaMallocManaged(reinterpret_cast<void**>(&data_buffer_), block_size_ * num_original_blocks_, cudaMemAttachHost);
  if (err != cudaSuccess) throw_error("Xorec (GPU Computation): Failed to allocate data buffer.");

  err = cudaMallocManaged(reinterpret_cast<void**>(&parity_buffer_), block_size_ * num_recovery_blocks_, cudaMemAttachHost);
  if (err != cudaSuccess) throw_error("Xorec (GPU Computation): Failed to allocate parity buffer.");

  block_bitmap_ = std::make_unique<uint8_t[]>(XOREC_MAX_TOTAL_BLOCKS);

  for (unsigned i = 0; i < num_original_blocks_; ++i) {
    int write_res = write_validation_pattern(i, &data_buffer_[i * block_size_], block_size_);
    if (write_res) throw_error("Xorec (GPU Computation): Failed to write random checking packet.");
  }
}