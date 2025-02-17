#include "baseline_benchmark.h"

int BaselineBenchmark::setup() {
  return -1;
}



void BaselineBenchmark::teardown() {
  return;
}



int BaselineBenchmark::encode() {
  return -1;
}



int BaselineBenchmark::decode() {
  return -1;
}



void BaselineBenchmark::flush_cache() {
  // TODO: Implement cache flushing
}



void BaselineBenchmark::check_for_corruption() {
  // TODO: Implement corruption checking
}



void BaselineBenchmark::simulate_data_loss() {
  // TODO: Implement data loss simulation
}


// int BaselineBenchmark::xor_function(uint8_t* dst, const uint8_t* src, size_t size) {
// #ifdef BASELINE_BENCHMARK_H
//   // If AVX2 is available, use it
//   size_t i = 0;

//   for (; i + 32 <= size; i += 32) {
//     __m256i a = _mm256_loadu_si256((__m256i*)&dst[i]);
//     __m256i b = _mm256_loadu_si256((__m256i*)&src[i]);
//     __m256i result = _mm256_xor_si256(a, b);
//     _mm256_storeu_si256((__m256i*)&dst[i], result);
//   }

//   for (; i < size; i++) {
//     dst[i] ^= src[i];
//   }

//   return 0;
// #else
//   for (size_t i = 0; i < size; i++) {
//     dst[i] ^= src[i];
//   }

//   return 0;
// #endif

// }


BaselineECC::BaselineECCResult BaselineECC::encode(
  size_t buffer_size,
  size_t original_count,
  size_t recovery_count,
  const uint8_t** original_data,
  uint8_t** recovery_data
) {
  if (buffer_size <= 0 || buffer_size % BASELINE_ECC_BLOCK_SIZE_ALIGNMENT != 0) {
    return BaselineECC_InvalidSize;
  }

  if (recovery_count <= 0 || recovery_count > original_count) {
    return BaselineECC_InvalidCounts;
  }

  if (!original_data || !recovery_data) {
    return BaselineECC_InvalidInput;
  }

  // Set recovery data to 0
  for (size_t i = 0; i < recovery_count; i++) {
    memset(recovery_data[i], 0, buffer_size);
  }

  // Generate recovery blocks using XOR

  for (size_t i = 0; i < original_count; i++) {
    for (size_t j = 0; j < recovery_count; j++) {
      if ((j+1) & (1 << i)) { // bitmask to select proper subsets
        xor_blocks(recovery_data[j*buffer_size], original_data[i*buffer_size], buffer_size);
      }
    }
  }

  return BaselineECC_Success;
}


BaselineECC::BaselineECCResult BaselineECC::decode(
  size_t buffer_size,
  size_t original_count,
  size_t recovery_count,
  uint8_t** output_data,
  const uint8_t** recovery_data,
  const std::vector<size_t>& lost_indices
) {
  if (buffer_size <= 0 || buffer_size % BASELINE_ECC_BLOCK_SIZE_ALIGNMENT != 0) {
    return BaselineECC_InvalidSize;
  }

  if (recovery_count <= 0 || recovery_count > original_count) {
    return BaselineECC_InvalidCounts;
  }

  if (!output_data || !recovery_data) {
    return BaselineECC_InvalidInput;
  }

  if (lost_indices.size() > recovery_count) {
    return BaselineECC_TooMuchData;
  }

  for (size_t k = 0; k < lost_indices.size(); k++) {
    size_t lost_index = lost_indices[k];

    if (lost_index >= original_count) {
      return BaselineECC_InvalidInput;
    }

    // Initialize the block to be recovered to 0
    memset(output_data[lost_index], 0, buffer_size);

    // XOR all the necessary recovery blocks
    for (size_t j = 0; j < recovery_count; j++) {
      if ((lost_index+1) & (1 << j)) {
        xor_blocks(output_data[lost_index * buffer_size], recovery_data[j * buffer_size], buffer_size);
      }
    }

    // XOR all the original blocks that were part of the recovery
    for (size_t i = 0; i < original_count; i++) {
      if (((lost_index+1) & (1 << i)) && i != lost_index) {
        xor_blocks(output_data[lost_index * buffer_size], output_data[i * buffer_size], buffer_size);
      }
    }
  }
  // TODO: Check for correctness
  return BaselineECC_Success;
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
