#ifndef UTILS_H
#define UTILS_H

#include <cstddef> // for size_t
#include <cstdint>



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