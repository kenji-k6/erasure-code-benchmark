#ifndef BASELINE_BENCHMARK_H
#define BASELINE_BENCHMARK_H

#include "abstract_benchmark.h"
#include "utils.h"

#include <immintrin.h> // AVX2
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <vector>
#include <memory>
#include <cstring>


class BaselineBenchmark : public ECCBenchmark {
public:
  int setup() override;
  void teardown() override;
  int encode() override;
  int decode() override;
  void flush_cache() override;
  void check_for_corruption() override;
  void simulate_data_loss() override;
  

private:
  int temp;
}; // class BaselineBenchmark



/*
 * Custom ECC implementation, used as a baseline for comparison
 * Uses a variation of Hamming Codes to encode and decode data
 * Can correct only a single error
*/
namespace BaselineECC{

  typedef enum BaselineECCResultT {
    BaselineECC_Success = 0,            // Operation succeeded
    BaselineECC_NeedMoreData = -1,      // Not enough recovery data received
    BaselineECC_TooMuchData = -2,       // Buffer counts are too high
    BaselineECC_InvalidSize = -3,       // Buffer size must be a multiple of 64 bytes
    BaselineECC_InvalidCounts = -4,     // Invalid counts provided
    BaselineECC_InvalidInput = -5,      // A function parameter was invalid
  } BaselineECCResult;


  BaselineECCResult encode(
    size_t buffer_size,                 // Number of bytes in each data buffer
    size_t original_count,              // Number of original_data[] buffer pointers
    size_t recovery_count,              // Number of recovery_data[] buffer pointers
    const uint8_t** original_data,      // Array of original data buffers
    uint8_t** recovery_data             // Array of recovery data buffers
  );

  BaselineECCResult decode(
    size_t buffer_size,                       // Number of bytes in each data buffer
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