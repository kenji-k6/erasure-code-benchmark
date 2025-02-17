#ifndef ABSTRACT_BENCHMARK_H
#define ABSTRACT_BENCHMARK_H

#include <cstddef>


/*
 * ECCBenchmark: Interface that all ECC libraries will implement
*/
class ECCBenchmark {
public:
  virtual ~ECCBenchmark() = default;

  // Initialize the benchmark with the given configuration
  virtual int setup() = 0;

  // Cleanup the benchmark
  virtual void teardown() = 0;

  // Run the encoding process
  virtual int encode() = 0;

  // Run the decoding process (with simulated data loss)
  virtual int decode() = 0;

  // Simulate a cold cache
  virtual void flush_cache() = 0;

  // Check for corruption in the decoded data
  virtual bool check_for_corruption() = 0;

  // Simulate data loss / corruption
  virtual void simulate_data_loss() = 0;

}; // class ECCBenchmark

#endif // ABSTRACT_BENCHMARK_H