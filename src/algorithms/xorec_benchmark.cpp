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
#include <cstring>


XORECBenchmark::XORECBenchmark(const BenchmarkConfig& config) noexcept : ECBenchmark(config) {}

int XORECBenchmark::setup() noexcept {
  // Allocate buffers with proper alignment for SIMD
  data_buffer_ = static_cast<uint8_t*>(aligned_alloc(ALIGNMENT_BYTES, block_size_ * num_original_blocks_));
  parity_buffer_ = static_cast<uint8_t*>(aligned_alloc(ALIGNMENT_BYTES, block_size_ * num_recovery_blocks_));

  if (!data_buffer_ || !parity_buffer_) {
    std::cerr << "XOREC: Failed to allocate buffer(s).\n";
    teardown();
    return -1;
  }
  
  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < num_original_blocks_; ++i) {
    int write_res = write_validation_pattern(i, data_buffer_ + i * block_size_, block_size_);
    if (write_res) {
      std::cerr << "XOREC: Failed to write random checking packet.\n";
      teardown();
      return -1;
    }
  }
  
  return 0;
}

void XORECBenchmark::teardown() noexcept {
  if (data_buffer_) free(data_buffer_);
  if (parity_buffer_) free(parity_buffer_);
}

int XORECBenchmark::encode() noexcept {
  xor_encode(data_buffer_, parity_buffer_, block_size_, num_original_blocks_, num_recovery_blocks_);
  return 0;
}

int XORECBenchmark::decode() noexcept {
  xor_decode(data_buffer_, parity_buffer_, block_size_, num_original_blocks_, num_recovery_blocks_, block_bitmap_);
  return 0;
}

void XORECBenchmark::simulate_data_loss() noexcept {
  uint32_t loss_idx = 0;
  for (unsigned i = 0; i < num_original_blocks_ + num_recovery_blocks_; ++i) {
    if (loss_idx < num_lost_blocks_ && lost_block_idxs_[loss_idx] == i) {
      if (i < num_original_blocks_) {
        memset(data_buffer_ + i * block_size_, 0, block_size_);
      } else {
        memset(parity_buffer_ + (i - num_original_blocks_) * block_size_, 0, block_size_);
      }

      ++loss_idx;
      continue;
    }
    block_bitmap_.set(i);
  }
}

bool XORECBenchmark::check_for_corruption() const noexcept {
  for (unsigned i = 0; i < num_original_blocks_; ++i) {
    if (!validate_block(data_buffer_ + i * block_size_, block_size_)) return false;
  }
  return true;
}



XORECScalarBenchmark::XORECScalarBenchmark(const BenchmarkConfig& config) noexcept : XORECBenchmark(config) {}
int XORECScalarBenchmark::encode() noexcept {
  xor_encode(data_buffer_, parity_buffer_, block_size_, num_original_blocks_, num_recovery_blocks_, XORVersion::Scalar);
  return 0;
}
int XORECScalarBenchmark::decode() noexcept {
  xor_decode(data_buffer_, parity_buffer_, block_size_, num_original_blocks_, num_recovery_blocks_, block_bitmap_, XORVersion::Scalar);
  return 0;
}

XORECAVXBenchmark::XORECAVXBenchmark(const BenchmarkConfig& config) noexcept : XORECBenchmark(config) {}
int XORECAVXBenchmark::encode() noexcept {
  xor_encode(data_buffer_, parity_buffer_, block_size_, num_original_blocks_, num_recovery_blocks_, XORVersion::AVX);
  return 0;
}
int XORECAVXBenchmark::decode() noexcept {
  xor_decode(data_buffer_, parity_buffer_, block_size_, num_original_blocks_, num_recovery_blocks_, block_bitmap_, XORVersion::AVX);
  return 0;
}

XORECAVX2Benchmark::XORECAVX2Benchmark(const BenchmarkConfig& config) noexcept : XORECBenchmark(config) {}
int XORECAVX2Benchmark::encode() noexcept {
  xor_encode(data_buffer_, parity_buffer_, block_size_, num_original_blocks_, num_recovery_blocks_, XORVersion::AVX2);
  return 0;
}
int XORECAVX2Benchmark::decode() noexcept {
  xor_decode(data_buffer_, parity_buffer_, block_size_, num_original_blocks_, num_recovery_blocks_, block_bitmap_, XORVersion::AVX2);
  return 0;
}



XORECBenchmarkGPU::XORECBenchmarkGPU(const BenchmarkConfig& config) noexcept : XORECBenchmark(config) {}

int XORECBenchmarkGPU::setup() noexcept {
  // Allocate
  cudaError_t err = aligned_cudaMallocManaged(reinterpret_cast<void**>(&data_buffer_), block_size_ * num_original_blocks_, ALIGNMENT_BYTES, cudaMemAttachHost);
  if (err != cudaSuccess) {
    std::cerr << "XOREC: Failed to allocate data buffer.\n";
    return -1;
  }

  parity_buffer_ = static_cast<uint8_t*>(aligned_alloc(ALIGNMENT_BYTES, block_size_ * num_recovery_blocks_));
  if (!parity_buffer_) {
    std::cerr << "XOREC: Failed to allocate parity buffer.\n";
    teardown();
    return -1;
  }

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < num_original_blocks_; ++i) {
    int write_res = write_validation_pattern(i, data_buffer_ + i * block_size_, block_size_);
    if (write_res) {
      std::cerr << "XOREC: Failed to write random checking packet.\n";
      teardown();
      return -1;
    }
  }
  return 0;
}

void XORECBenchmarkGPU::teardown() noexcept {
  if (data_buffer_) aligned_cudaFree(data_buffer_);
  if (parity_buffer_) free(parity_buffer_);
}

void XORECBenchmarkGPU::make_memory_cold() noexcept {
  touch_memory(data_buffer_, block_size_ * num_original_blocks_);
}



XORECScalarBenchmarkGPU::XORECScalarBenchmarkGPU(const BenchmarkConfig& config) noexcept : XORECBenchmarkGPU(config) {}
int XORECScalarBenchmarkGPU::encode() noexcept {
  xor_encode(data_buffer_, parity_buffer_, block_size_, num_original_blocks_, num_recovery_blocks_, XORVersion::Scalar);
  return 0;
}
int XORECScalarBenchmarkGPU::decode() noexcept {
  xor_decode(data_buffer_, parity_buffer_, block_size_, num_original_blocks_, num_recovery_blocks_, block_bitmap_, XORVersion::Scalar);
  return 0;
}

XORECAVXBenchmarkGPU::XORECAVXBenchmarkGPU(const BenchmarkConfig& config) noexcept : XORECBenchmarkGPU(config) {}
int XORECAVXBenchmarkGPU::encode() noexcept {
  xor_encode(data_buffer_, parity_buffer_, block_size_, num_original_blocks_, num_recovery_blocks_, XORVersion::AVX);
  return 0;
}
int XORECAVXBenchmarkGPU::decode() noexcept {
  xor_decode(data_buffer_, parity_buffer_, block_size_, num_original_blocks_, num_recovery_blocks_, block_bitmap_, XORVersion::AVX);
  return 0;
}

XORECAVX2BenchmarkGPU::XORECAVX2BenchmarkGPU(const BenchmarkConfig& config) noexcept : XORECBenchmarkGPU(config) {}
int XORECAVX2BenchmarkGPU::encode() noexcept {
  xor_encode(data_buffer_, parity_buffer_, block_size_, num_original_blocks_, num_recovery_blocks_, XORVersion::AVX2);
  return 0;
}
int XORECAVX2BenchmarkGPU::decode() noexcept {
  xor_decode(data_buffer_, parity_buffer_, block_size_, num_original_blocks_, num_recovery_blocks_, block_bitmap_, XORVersion::AVX2);
  return 0;
}