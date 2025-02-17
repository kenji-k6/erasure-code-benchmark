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


typedef void (*xor_func_t)(uint8_t*, const uint8_t*, size_t);


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
*/
namespace BaselineECC{
  int encode(
    size_t buffer_size,                 // Number of bytes in each data buffer
    size_t original_count,              // Number of original_data[] buffer pointers
    size_t recovery_count,              // Number of recovery_data[] buffer pointers
    const uint8_t** original_data,      // Array of original data buffers
    uint8_t** recovery_data             // Array of recovery data buffers
  );

  int decode(
    size_t buffer_size,                 // Number of bytes in each data buffer
    size_t original_count,              // Number of original_data[] buffer pointers
    size_t recovery_count,              // Number of recovery_data[] buffer pointers
    uint8_t** original_data,            // Array of original data buffers
    const uint8_t** recovery_data       // Array of recovery data buffers
  );

  // Internal xor_function (uses AVX2 if available)
  void xor_function(
    uint8_t* dst,         // Destination buffer
    const uint8_t* src,   // Source buffer
    size_t size        // Number of bytes to xor (buffer size)
  );
}



#endif // BASELINE_BENCHMARK_H