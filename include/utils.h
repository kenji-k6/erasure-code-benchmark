#ifndef UTILS_H
#define UTILS_H

#include <cstddef> // for size_t
#include <cstdint>
#include <chrono>


#define ALIGNMENT_BYTES 16 // This ensures that allocs are 16-byte aligned, to allow SIMD instructions

// Compiler-specific force inline keyword
#ifdef _MSC_VER
    #define UTIL_FORCE_INLINE inline __forceinline
#else
    #define UTIL_FORCE_INLINE inline __attribute__((always_inline))
#endif

#define MEGABYTE_TO_BYTE_FACTOR 1000000



/*
 * PCGRandom number generator, used to generate random data for benchmarking
 * Implemented according to https://www.pcg-random.org/
*/

class PCGRandom {
private:
  uint64_t state; // Internal state
  uint64_t inc;   // Increment, *must be odd*

public:
  PCGRandom(uint64_t seed, uint64_t seq); // Constructor: provide a seed and a sequence number
  uint32_t next(); // Generate a random 32-bit number
}; // class PCGRandom





/*
 * Timer utility functions
*/
// Get the current time in microseconds
inline long long get_current_time_us() {
  auto now = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
  return duration.count();
}

/*
 * BenchmarkConfig: Configuration parameters for the benchmark
*/

struct BenchmarkConfig {
  // Common parameters
  size_t data_size;             // Total size of original data
  size_t block_size;            // Size of each block
  double redundancy_ratio;       // Recovery blocks / original blocks ratio
  double loss_rate;             // Simulated data loss rate
  int iterations;               // Number of iterations to run the benchmark

  struct {                      // Derived value (calculated during setup)
    size_t original_blocks;
    size_t recovery_blocks;
  } computed;
}; // struct BenchmarkConfig

/*
 * Functions to allocate and free aligned memory (to allow SIMD instructions)
*/
static UTIL_FORCE_INLINE void* simd_safe_allocate(size_t size);

static UTIL_FORCE_INLINE void simd_safe_free(void* ptr);


#endif // UTILS_H