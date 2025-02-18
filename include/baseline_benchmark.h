#ifndef BASELINE_BENCHMARK_H
#define BASELINE_BENCHMARK_H

#include "abstract_benchmark.h"
#include "utils.h"

#ifdef __AVX2__
#include <immintrin.h> // AVX2
#endif

#include <cstdint>
#include <cstddef>
#include <iostream>
#include <vector>
#include <memory>
#include <cstring>


class BaselineBenchmark : public ECCBenchmark {
public:
  int setup() noexcept override;
  void teardown() noexcept override;
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
  bool check_for_corruption() const noexcept override;
  void flush_cache() noexcept override;
  

private:
  uint8_t* data_; // recovery data will be after original data
  uint8_t** original_blocks_;
  uint8_t** recovery_blocks_;
  std::vector<size_t> lost_indices_;
}; // class BaselineBenchmark



/*
 * Custom ECC implementation, used as a baseline for comparison
 *
*/
namespace BaselineECC{

  int encode(
    size_t block_size,                  // Number of bytes in each data block
    size_t original_count,              // Number of original_data[] buffer pointers
    size_t recovery_count,              // Number of recovery_data[] buffer pointers
    const uint8_t** original_data,      // Array of original data buffers
    uint8_t** recovery_data             // Array of recovery data buffers
  );

  int decode(
    size_t block_size,                        // Number of bytes in each data block
    size_t original_count,                    // Number of original_data[] buffer pointers
    size_t recovery_count,                    // Number of recovery_data[] buffer pointers
    uint8_t** output_data,                    // Array of original data buffers (this is where output will go to)
    const uint8_t** recovery_data,            // Array of recovery data buffers
    const std::vector<size_t>& lost_indices   // Indices of lost data
  );

  // Internal xor_function (uses AVX2 if available)
  void xor_blocks(
    uint8_t* dst,         // Destination buffer
    const uint8_t* src,   // Source buffer
    size_t size        // Number of bytes to xor (buffer size)
  );
}



#endif // BASELINE_BENCHMARK_H