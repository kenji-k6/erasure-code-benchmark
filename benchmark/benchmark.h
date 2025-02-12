#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "aff3ct.hpp"
#include "cm256.h"
#include "erasure_code.h" // ISA-L
#include "leopard.h"
#include "wirehair/wirehair.h"

#include "utils.h"

#include <iostream>


#define LEOPARD_MIN_BLOCKS 2
#define LEOPARD_MAX_BLOCKS 65536
#define LEOPARD_BLOCK_SIZE_ALIGNMENT 64

#define WIREHAIR_MIN_BLOCKS 2
#define WIREHAIR_MAX_BLOCKS 64000


/*
 * BenchmarkConfig: Configuration parameters for the benchmark
*/

struct BenchmarkConfig {
  // Common parameters
  size_t data_size;             // Total size of original data
  size_t block_size;            // Size of each block
  float redudandy_ratio;        // Recovery blocks / original blocks ratio
  int iterations;               // Number of iterations to run the benchmark

  struct {                      // Derived value (calculated during setup)
    size_t original_blocks;
    size_t recovery_blocks;
    size_t actual_block_size; 
  } computed;
}; // struct BenchmarkConfig


/*
 * ECCBenchmark: Interface that all ECC libraries will implement
*/
class ECCBenchmark {
public:
  virtual ~ECCBenchmark() = default;

  // Initialize the benchmark with the given configuration
  virtual int setup(const BenchmarkConfig& config) = 0;

  // Run the encoding process
  virtual int encode() = 0;

  // Run the decoding process (with simulated data loss)
  virtual int decode(float loss_rate) = 0;

  // Cleanup the benchmark
  virtual void teardown() = 0;

  // Metrics collected during the benchmark
  struct Metrics {
    long long encode_time_us;
    long long decode_time_us;
    size_t memory_usage;
    double throughput_mbps;
  };

  // Get the metrics collected during the benchmark
  virtual Metrics get_metrics() = 0;
}; // class ECCBenchmark




/*
 * BenchmarkRunner: Main class that runs the benchmark
*/
class BenchmarkRunner {
public:
  enum class Library {
    aff3ct,
    cm256,
    isa_l,
    leopard,
    wirehair
  };

  // Add a test case to the runner
  void add_test_case(Library lib, const BenchmarkConfig& config);

  // Run all the test cases
  void run_all();

private:
  // Helper to create a benchmark instance for a specific library
  std::unique_ptr<ECCBenchmark> create_benchmark(Library lib);

  // Run a single test case
  void run_single(ECCBenchmark& bench, const BenchmarkConfig& config);

  // Save results to a CSV file
  void save_results(Library lib, const BenchmarkConfig& config, const ECCBenchmark::Metrics& metrics);

  // Computes the left over configuration parameters 
  void compute_config(BenchmarkConfig& config);

  // Collection of test cases
  std::vector<std::pair<Library, BenchmarkConfig>> test_cases_;
};

#endif