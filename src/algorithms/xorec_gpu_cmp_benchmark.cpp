#include "xorec_gpu_cmp_benchmark.hpp"
#include "xorec_gpu_cmp.cuh"
#include "cuda_utils.cuh"
#include "utils.hpp"
#include <cstring>

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

XorecBenchmarkGPUCmp::~XorecBenchmarkGPUCmp() noexcept {
  if (data_buffer_) cudaFree(data_buffer_);
  if (parity_buffer_) cudaFree(parity_buffer_);
}

int XorecBenchmarkGPUCmp::encode() noexcept {
  xorec_gpu_encode(data_buffer_, parity_buffer_, block_size_, num_original_blocks_, num_recovery_blocks_);
  return 0;
}

int XorecBenchmarkGPUCmp::decode() noexcept {
  xorec_gpu_decode(data_buffer_, parity_buffer_, block_size_, num_original_blocks_, num_recovery_blocks_, block_bitmap_.get());
  return 0;
}

void XorecBenchmarkGPUCmp::simulate_data_loss() noexcept {
  uint32_t loss_idx = 0;
  for (unsigned i = 0; i < num_total_blocks_; ++i) {
    if (loss_idx < num_lost_blocks_ && lost_block_idxs_[loss_idx] == i) {
      if (i < num_original_blocks_) {
        memset(data_buffer_ + i * block_size_, 0, block_size_);
      } else {
        memset(&parity_buffer_[(i - num_original_blocks_) * block_size_], 0, block_size_);
      }

      ++loss_idx;
      continue;
    }
    if (i < num_original_blocks_) {
      block_bitmap_[i] = 1;
    } else {
      block_bitmap_[XOREC_MAX_DATA_BLOCKS + i] = 1;
    }
  }
}

bool XorecBenchmarkGPUCmp::check_for_corruption() const noexcept {
  for (unsigned i = 0; i < num_original_blocks_; ++i) {
    if (!validate_block(&data_buffer_[i * block_size_], block_size_)) return false;
  }
  return true;
}

void XorecBenchmarkGPUCmp::touch_gpu_memory() noexcept {
  touch_memory(data_buffer_, block_size_ * num_original_blocks_);
  touch_memory(parity_buffer_, block_size_ * num_recovery_blocks_);
}
