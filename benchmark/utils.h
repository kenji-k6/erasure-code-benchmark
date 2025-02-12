#ifndef UTILS_H
#define UTILS_H

#include <cstddef> // for size_t
#include <cstdint>
#include <chrono>



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

#endif // UTILS_H



/*
 * Timer utility functions
*/
// Get the current time in microseconds
inline long long get_current_time_us() {
  auto now = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
  return duration.count();
}

