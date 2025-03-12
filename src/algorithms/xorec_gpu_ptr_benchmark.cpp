
#include "xorec.h"
#include "xorec_gpu_ptr_benchmark.h"
#include "cuda_utils.cuh"
#include "utils.h"

XorecBenchmarkGPUPtr::XorecBenchmarkGPUPtr(const BenchmarkConfig& config) noexcept : ECBenchmark(config) {
  num_total_blocks_ = num_original_blocks_ + num_recovery_blocks_;
  cudaError_t err = cudaMallocManaged(reinterpret_cast<void**>(&data_buffer_), block_size_ * num_original_blocks_, cudaMemAttachHost); 
  if (err != cudaSuccess) throw_error("Xorec: Failed to allocate data buffer.");

  parity_buffer_ = std::make_unique<uint8_t[]>(block_size_ * num_recovery_blocks_);
  if (!parity_buffer_) throw_error("Xorec: Failed to allocate parity buffer.");

  block_bitmap_ = std::make_unique<uint8_t[]>(XOREC_MAX_TOTAL_BLOCKS);

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < num_original_blocks_; ++i) {
    int write_res = write_validation_pattern(i, &data_buffer_[i * block_size_], block_size_);
    if (write_res) throw_error("Xorec: Failed to write random checking packet.");
  }
}

XorecBenchmarkGPUPtr::~XorecBenchmarkGPUPtr() noexcept {
  if (data_buffer_) cudaFree(data_buffer_);
}

void XorecBenchmarkGPUPtr::simulate_data_loss() noexcept {
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
      block_bitmap_[128 + i] = 1;
    }
  }
}

bool XorecBenchmarkGPUPtr::check_for_corruption() const noexcept {
  for (unsigned i = 0; i < num_original_blocks_; ++i) {
    if (!validate_block(&data_buffer_[i * block_size_], block_size_)) return false;
  }
  return true;
}

void XorecBenchmarkGPUPtr::touch_gpu_memory() noexcept {
  touch_memory(data_buffer_, block_size_ * num_original_blocks_);
}




XorecBenchmarkScalarGPUPtr::XorecBenchmarkScalarGPUPtr(const BenchmarkConfig& config) noexcept : XorecBenchmarkGPUPtr(config) {}
int XorecBenchmarkScalarGPUPtr::encode() noexcept {
  xorec_encode(data_buffer_, parity_buffer_.get(), block_size_, num_original_blocks_, num_recovery_blocks_, XorecVersion::Scalar);
  return 0;
}
int XorecBenchmarkScalarGPUPtr::decode() noexcept {
  xorec_decode(data_buffer_, parity_buffer_.get(), block_size_, num_original_blocks_, num_recovery_blocks_, block_bitmap_.get(), XorecVersion::Scalar);
  return 0;
}

XorecBenchmarkAVXGPUPtr::XorecBenchmarkAVXGPUPtr(const BenchmarkConfig& config) noexcept : XorecBenchmarkGPUPtr(config) {}
int XorecBenchmarkAVXGPUPtr::encode() noexcept {
  xorec_encode(data_buffer_, parity_buffer_.get(), block_size_, num_original_blocks_, num_recovery_blocks_, XorecVersion::AVX);
  return 0;
}
int XorecBenchmarkAVXGPUPtr::decode() noexcept {
  xorec_decode(data_buffer_, parity_buffer_.get(), block_size_, num_original_blocks_, num_recovery_blocks_, block_bitmap_.get(), XorecVersion::AVX);
  return 0;
}

XorecBenchmarkAVX2GPUPtr::XorecBenchmarkAVX2GPUPtr(const BenchmarkConfig& config) noexcept : XorecBenchmarkGPUPtr(config) {}
int XorecBenchmarkAVX2GPUPtr::encode() noexcept {
  xorec_encode(data_buffer_, parity_buffer_.get(), block_size_, num_original_blocks_, num_recovery_blocks_, XorecVersion::AVX2);
  return 0;
}
int XorecBenchmarkAVX2GPUPtr::decode() noexcept {
  xorec_decode(data_buffer_, parity_buffer_.get(), block_size_, num_original_blocks_, num_recovery_blocks_, block_bitmap_.get(), XorecVersion::AVX2);
  return 0;
}