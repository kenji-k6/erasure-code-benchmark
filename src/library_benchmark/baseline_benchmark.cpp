#include "baseline_benchmark.h"

int BaselineBenchmark::setup() noexcept {
  // Allocate buffers
  data_ = (uint8_t*) aligned_alloc(ALIGNMENT_BYTES, (benchmark_config.computed.num_original_blocks + benchmark_config.computed.num_recovery_blocks) * benchmark_config.block_size);

  if (!data_) {
    teardown();
    std::cerr << "Baseline: Failed to allocate data buffer.\n";
    return -1;
  }
  
  // Allocate pointers
  original_blocks_ = new uint8_t*[benchmark_config.computed.num_original_blocks];
  recovery_blocks_ = new uint8_t*[benchmark_config.computed.num_recovery_blocks];

  if (!original_blocks_ || !recovery_blocks_) {
    teardown();
    std::cerr << "Baseline: Failed to allocate pointer arrays.\n";
    return -1;
  }

  

  // Initialize pointers
  for (size_t i = 0; i < benchmark_config.computed.num_original_blocks; i++) {
    original_blocks_[i] = data_ + i * benchmark_config.block_size;
  }
  
  for (size_t i = 0; i < benchmark_config.computed.num_recovery_blocks; i++) {
    recovery_blocks_[i] = data_ + (benchmark_config.computed.num_original_blocks + i) * benchmark_config.block_size;
  }

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < benchmark_config.computed.num_original_blocks; i++) {
    int write_res = write_validation_pattern(
      i,
      original_blocks_[i],
      benchmark_config.block_size
    );

    if (write_res) {
      teardown();
      std::cerr << "Baseline: Failed to write random checking packet.\n";
      return -1;
    }
  }

  // TODO: Change the lost indices appropriately
  // lost_indices_.resize(2);
  // lost_indices_[0] = 0;
  // lost_indices_[1] = 1;

  return 0;
}



void BaselineBenchmark::teardown() noexcept {
  if (data_) free(data_);
  if (original_blocks_) delete[] original_blocks_;
  if (recovery_blocks_) delete[] recovery_blocks_;
}



int BaselineBenchmark::encode() noexcept {
  return BaselineECC::encode(
    benchmark_config.block_size,
    benchmark_config.computed.num_original_blocks,
    benchmark_config.computed.num_recovery_blocks,
    (const_cast<const uint8_t**>(original_blocks_)),
    recovery_blocks_
  );
}



int BaselineBenchmark::decode() noexcept {
  return BaselineECC::decode(
    benchmark_config.block_size,
    benchmark_config.computed.num_original_blocks,
    benchmark_config.computed.num_recovery_blocks,
    original_blocks_,
    (const_cast<const uint8_t**>(recovery_blocks_)),
    lost_indices_
  );
}



void BaselineBenchmark::flush_cache() noexcept {
  // TODO: Implement cache flushing
}



bool BaselineBenchmark::check_for_corruption() const noexcept {
  return true;
}



void BaselineBenchmark::simulate_data_loss() noexcept {
  // TODO: Implement data loss simulation
}


int BaselineECC::encode(
  size_t block_size,
  size_t original_count,
  size_t recovery_count,
  const uint8_t** original_data,
  uint8_t** recovery_data
) {
  if (block_size <= 0 || block_size % ECCLimits::BASELINE_BLOCK_ALIGNMENT != 0) {
    return -1;
  }

  if (recovery_count <= 0 || recovery_count > original_count) {
    return -1;
  }

  if (!original_data || !recovery_data) {
    return -1;
  }

  // Set recovery data to 0
  for (size_t i = 0; i < recovery_count; i++) {
    memset(recovery_data[i], 0, block_size);
  }

  // Generate recovery blocks using XOR
  for (size_t i = 0; i < original_count; i++) {
    for (size_t j = 0; j < recovery_count; j++) {
      if ((j+1) & (1 << i)) { // bitmask to select proper subsets
        xor_blocks(recovery_data[j], original_data[i], block_size);
      }
    }
  }

  return 0;
}


int BaselineECC::decode(
  size_t block_size,
  size_t original_count,
  size_t recovery_count,
  uint8_t** output_data,
  const uint8_t** recovery_data,
  const std::vector<size_t>& lost_indices
) {
  if (block_size <= 0 || block_size % ECCLimits::BASELINE_BLOCK_ALIGNMENT != 0) {
    return -1;
  }

  if (recovery_count <= 0 || recovery_count > original_count) {
    return -1;
  }

  if (!output_data || !recovery_data) {
    return -1;
  }

  if (lost_indices.size() > recovery_count) {
    return -1;
  }

  for (size_t k = 0; k < lost_indices.size(); k++) {
    size_t lost_index = lost_indices[k];

    if (lost_index >= original_count) {
      return -1;
    }

    // Initialize the block to be recovered to 0
    memset(output_data[lost_index], 0, block_size);

    // XOR all the necessary recovery blocks
    for (size_t j = 0; j < recovery_count; j++) {
      if ((lost_index+1) & (1 << j)) {
        xor_blocks(output_data[lost_index], recovery_data[j], block_size);
      }
    }

    // XOR all the original blocks that were part of the recovery
    for (size_t i = 0; i < original_count; i++) {
      if (((lost_index+1) & (1 << i)) && i != lost_index) {
        xor_blocks(output_data[lost_index], output_data[i], block_size);
      }
    }
  }
  // TODO: Check for correctness
  return 0;
}



void BaselineECC::xor_blocks(
  uint8_t* dst,
  const uint8_t* src,
  size_t size
) {
#ifdef __AVX2__
  size_t i = 0;

  for (; i + 32 <= size; i += 32) {
    __m256i a = _mm256_loadu_si256((__m256i*)&dst[i]);
    __m256i b = _mm256_loadu_si256((__m256i*)&src[i]);
    __m256i result = _mm256_xor_si256(a, b);
    _mm256_storeu_si256((__m256i*)&dst[i], result);
  }

  for (; i < size; i++) {
    dst[i] ^= src[i];
  }

#else
  for (size_t i = 0; i < size; i++) {
    dst[i] ^= src[i];
  }
#endif
}
