#include "gpu_xorec.cuh"
#include "all_gpu_benchmark.h"
#include "utils.h"
#include "cuda_utils.cuh"
#include <cstring>

GPUBenchmark::GPUBenchmark(const BenchmarkConfig& config) noexcept : ECBenchmark(config) {
  num_total_blocks_ = num_original_blocks_ + num_recovery_blocks_;
  cudaError_t err = cudaMallocManaged(reinterpret_cast<void**>(&data_buffer_), block_size_ * num_original_blocks_, cudaMemAttachHost);
  if (err != cudaSuccess) throw_error("Failed to allocate memory for data buffer: " + std::string(cudaGetErrorString(err)));


  err = cudaMallocManaged(reinterpret_cast<void**>(&parity_buffer_), block_size_ * num_recovery_blocks_, cudaMemAttachHost);
  if (err != cudaSuccess) throw_error("Failed to allocate memory for parity buffer: " + std::string(cudaGetErrorString(err)));

  for (unsigned i = 0; i < num_original_blocks_; ++i) {
    int write_res = write_validation_pattern(i, &data_buffer_[i * block_size_], block_size_);
    if (write_res) throw_error("XOREC: Failed to write random checking packet.");
  }
}


GPUBenchmark::~GPUBenchmark() noexcept {
  if (data_buffer_) cudaFree(data_buffer_);
  if (parity_buffer_) cudaFree(parity_buffer_);
}

int GPUBenchmark::encode() noexcept {
  XorecGPU::xor_encode(data_buffer_, parity_buffer_, block_size_, num_original_blocks_, num_recovery_blocks_);
  return 0;
}

int GPUBenchmark::decode() noexcept {
  XorecGPU::xor_decode(data_buffer_, parity_buffer_, block_size_, num_original_blocks_, num_recovery_blocks_, block_bitmap_);
  return 0;
}

void GPUBenchmark::simulate_data_loss() noexcept {
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
    block_bitmap_.set(i);
  }
}

bool GPUBenchmark::check_for_corruption() const noexcept {
  for (unsigned i = 0; i < num_original_blocks_; ++i) {
    if (!validate_block(&data_buffer_[i * block_size_], block_size_)) return false;
  }
  return true;
}

void GPUBenchmark::make_memory_cold() noexcept {
  touch_memory(data_buffer_, block_size_ * num_original_blocks_);
  touch_memory(parity_buffer_, block_size_ * num_recovery_blocks_);
}