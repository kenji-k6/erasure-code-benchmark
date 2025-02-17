#ifndef UTILS_H
#define UTILS_H

#include <cstddef> // for size_t
#include <cstdint>
#include <chrono>

// This ensures that allocs are 64-byte aligned, to allow SIMD instructions, also needed by ISA-L
#define ALIGNMENT_BYTES 64

// Compiler-specific force inline keyword
#ifdef _MSC_VER
    #define UTIL_FORCE_INLINE inline __forceinline
#else
    #define UTIL_FORCE_INLINE inline __attribute__((always_inline))
#endif

#define MEGABYTE_TO_BYTE_FACTOR 1000000


#define LEOPARD_MIN_ORIG_BLOCKS 2
#define LEOPARD_MAX_ORIG_BLOCKS 65536
#define LEOPARD_BLOCK_SIZE_ALIGNMENT 64

#define CM256_MIN_ORIG_BLOCKS 1
#define CM256_MAX_ORIG_BLOCKS 256

#define WIREHAIR_MIN_ORIG_BLOCKS 2
#define WIREHAIR_MAX_ORIG_BLOCKS 64000

#define ISAL_MIN_BLOCK_SIZE 64
#define ISAL_MAX_ORIG_BLOCKS 255
#define ISAL_MAX_TOT_BLOCKS 255

#define BASELINE_ECC_BLOCK_SIZE_ALIGNMENT 64


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

extern BenchmarkConfig kConfig;



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
 * Functions to allocate and free aligned memory (to allow SIMD instructions)
*/
static UTIL_FORCE_INLINE void* simd_safe_allocate(size_t size) {
  uint8_t* data = (uint8_t *) calloc(1, ALIGNMENT_BYTES + size);

  if (!data) return nullptr;
  
  unsigned offset = (unsigned)((uintptr_t)data % ALIGNMENT_BYTES);
  data += ALIGNMENT_BYTES - offset;
  data[-1] = (uint8_t)offset;
  return (void*)data;
}

static UTIL_FORCE_INLINE void simd_safe_free(void* ptr) {
  if (!ptr) return;
  uint8_t* data = (uint8_t*)ptr;
  unsigned offset = data[-1];
  if (offset >= ALIGNMENT_BYTES) exit(1); // should never happen
  data -= ALIGNMENT_BYTES - offset;
  free(data);
}

#endif // UTILS_H