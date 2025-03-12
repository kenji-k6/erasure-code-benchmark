/**
 * @file xorec_benchmark.cpp
 * @brief Benchmark implementation for the XOR-EC implementation.
 * 
 * Documentation can be found in xorec.h and abstract_benchmark.h
 */


#include "xorec_benchmark.h"
#include "xorec.h"
#include "cuda_utils.cuh"
#include "utils.h"
#include "xorec_gpu.cuh"
#include <cstring>


XorecBenchmark::XorecBenchmark(const BenchmarkConfig& config) noexcept : ECBenchmark(config) {
  num_total_blocks_ = num_original_blocks_ + num_recovery_blocks_;
  data_buffer_ = std::make_unique<uint8_t[]>(block_size_ * num_original_blocks_);
  parity_buffer_ = std::make_unique<uint8_t[]>(block_size_ * num_recovery_blocks_);
  block_bitmap_ = std::make_unique<uint8_t[]>(XOREC_MAX_TOTAL_BLOCKS);

  if (!data_buffer_ || !parity_buffer_) throw_error("Xorec: Failed to allocate buffer(s).");

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < num_original_blocks_; ++i) {
    int write_res = write_validation_pattern(i, &data_buffer_[i * block_size_], block_size_);
    if (write_res) throw_error("Xorec: Failed to write random checking packet.");
  }
}

int XorecBenchmark::encode() noexcept {
  xorec_encode(data_buffer_.get(), parity_buffer_.get(), block_size_, num_original_blocks_, num_recovery_blocks_);
  return 0;
}

int XorecBenchmark::decode() noexcept {
  xorec_decode(data_buffer_.get(), parity_buffer_.get(), block_size_, num_original_blocks_, num_recovery_blocks_, block_bitmap_.get());
  return 0;
}

void XorecBenchmark::simulate_data_loss() noexcept {
  uint32_t loss_idx = 0;
  for (unsigned i = 0; i < num_total_blocks_; ++i) {
    if (loss_idx < num_lost_blocks_ && lost_block_idxs_[loss_idx] == i) {
      if (i < num_original_blocks_) {
        memset(&data_buffer_[i * block_size_], 0, block_size_);
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

bool XorecBenchmark::check_for_corruption() const noexcept {
  for (unsigned i = 0; i < num_original_blocks_; ++i) {
    if (!validate_block(&data_buffer_[i * block_size_], block_size_)) return false;
  }
  return true;
}


XorecBenchmarkScalar::XorecBenchmarkScalar(const BenchmarkConfig& config) noexcept : XorecBenchmark(config) {}
int XorecBenchmarkScalar::encode() noexcept {
  xorec_encode(data_buffer_.get(), parity_buffer_.get(), block_size_, num_original_blocks_, num_recovery_blocks_, XorecVersion::Scalar);
  return 0;
}
int XorecBenchmarkScalar::decode() noexcept {
  xorec_decode(data_buffer_.get(), parity_buffer_.get(), block_size_, num_original_blocks_, num_recovery_blocks_, block_bitmap_.get(), XorecVersion::Scalar);
  return 0;
}


XorecBenchmarkAVX::XorecBenchmarkAVX(const BenchmarkConfig& config) noexcept : XorecBenchmark(config) {}
int XorecBenchmarkAVX::encode() noexcept {
  xorec_encode(data_buffer_.get(), parity_buffer_.get(), block_size_, num_original_blocks_, num_recovery_blocks_, XorecVersion::AVX);
  return 0;
}
int XorecBenchmarkAVX::decode() noexcept {
  xorec_decode(data_buffer_.get(), parity_buffer_.get(), block_size_, num_original_blocks_, num_recovery_blocks_, block_bitmap_.get(), XorecVersion::AVX);
  return 0;
}

XorecBenchmarkAVX2::XorecBenchmarkAVX2(const BenchmarkConfig& config) noexcept : XorecBenchmark(config) {}
int XorecBenchmarkAVX2::encode() noexcept {
  xorec_encode(data_buffer_.get(), parity_buffer_.get(), block_size_, num_original_blocks_, num_recovery_blocks_, XorecVersion::AVX2);
  return 0;
}
int XorecBenchmarkAVX2::decode() noexcept {
  xorec_decode(data_buffer_.get(), parity_buffer_.get(), block_size_, num_original_blocks_, num_recovery_blocks_, block_bitmap_.get(), XorecVersion::AVX2);
  return 0;
}



XorecBenchmarkGPUPointer::XorecBenchmarkGPUPointer(const BenchmarkConfig& config) noexcept : ECBenchmark(config) {
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

XorecBenchmarkGPUPointer::~XorecBenchmarkGPUPointer() noexcept {
  if (data_buffer_) cudaFree(data_buffer_);
}

int XorecBenchmarkGPUPointer::encode() noexcept {
  xorec_encode(data_buffer_, parity_buffer_.get(), block_size_, num_original_blocks_, num_recovery_blocks_);
  return 0;
}

int XorecBenchmarkGPUPointer::decode() noexcept {
  xorec_decode(data_buffer_, parity_buffer_.get(), block_size_, num_original_blocks_, num_recovery_blocks_, block_bitmap_.get());
  return 0;
}

void XorecBenchmarkGPUPointer::simulate_data_loss() noexcept {
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

bool XorecBenchmarkGPUPointer::check_for_corruption() const noexcept {
  for (unsigned i = 0; i < num_original_blocks_; ++i) {
    if (!validate_block(&data_buffer_[i * block_size_], block_size_)) return false;
  }
  return true;
}

void XorecBenchmarkGPUPointer::touch_gpu_memory() noexcept {
  touch_memory(data_buffer_, block_size_ * num_original_blocks_);
}




XorecBenchmarkScalarGPUPointer::XorecBenchmarkScalarGPUPointer(const BenchmarkConfig& config) noexcept : XorecBenchmarkGPUPointer(config) {}
int XorecBenchmarkScalarGPUPointer::encode() noexcept {
  xorec_encode(data_buffer_, parity_buffer_.get(), block_size_, num_original_blocks_, num_recovery_blocks_, XorecVersion::Scalar);
  return 0;
}
int XorecBenchmarkScalarGPUPointer::decode() noexcept {
  xorec_decode(data_buffer_, parity_buffer_.get(), block_size_, num_original_blocks_, num_recovery_blocks_, block_bitmap_.get(), XorecVersion::Scalar);
  return 0;
}

XorecBenchmarkAVXGPUPointer::XorecBenchmarkAVXGPUPointer(const BenchmarkConfig& config) noexcept : XorecBenchmarkGPUPointer(config) {}
int XorecBenchmarkAVXGPUPointer::encode() noexcept {
  xorec_encode(data_buffer_, parity_buffer_.get(), block_size_, num_original_blocks_, num_recovery_blocks_, XorecVersion::AVX);
  return 0;
}
int XorecBenchmarkAVXGPUPointer::decode() noexcept {
  xorec_decode(data_buffer_, parity_buffer_.get(), block_size_, num_original_blocks_, num_recovery_blocks_, block_bitmap_.get(), XorecVersion::AVX);
  return 0;
}

XorecBenchmarkAVX2GPUPointer::XorecBenchmarkAVX2GPUPointer(const BenchmarkConfig& config) noexcept : XorecBenchmarkGPUPointer(config) {}
int XorecBenchmarkAVX2GPUPointer::encode() noexcept {
  xorec_encode(data_buffer_, parity_buffer_.get(), block_size_, num_original_blocks_, num_recovery_blocks_, XorecVersion::AVX2);
  return 0;
}
int XorecBenchmarkAVX2GPUPointer::decode() noexcept {
  xorec_decode(data_buffer_, parity_buffer_.get(), block_size_, num_original_blocks_, num_recovery_blocks_, block_bitmap_.get(), XorecVersion::AVX2);
  return 0;
}



XorecBenchmarkGPUComputation::XorecBenchmarkGPUComputation(const BenchmarkConfig& config) noexcept : ECBenchmark(config) {
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