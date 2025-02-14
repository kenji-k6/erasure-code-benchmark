#ifndef BENCHMARK_RUNNER_H
#define BENCHMARK_RUNNER_H

#include "abstract_benchmark.h"
#include "utils.h"

#include "cm256_benchmark.h"
#include "leopard_benchmark.h"
#include "wirehair_benchmark.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <stdexcept>


// // Assert that the block size is a multiple of 64 bytes
// if (config_.block_size % LEOPARD_BLOCK_SIZE_ALIGNMENT != 0) {
//   std::cerr << "Leopard: Block size must be a multiple of " << LEOPARD_BLOCK_SIZE_ALIGNMENT << " bytes.\n";
//   return -1;
// }

// // Assert that the number of blocks is within the valid range
// if (config_.computed.original_blocks < LEOPARD_MIN_BLOCKS || config_.computed.original_blocks > LEOPARD_MAX_BLOCKS) {
//   std::cerr << "Leopard: Original blocks must be between " << LEOPARD_MIN_BLOCKS << " and " << LEOPARD_MAX_BLOCKS << " (is " << config_.computed.original_blocks << ").\n";
//   return -1;
// }

// if (config_.computed.original_blocks < CM256_MIN_BLOCKS || config.computed.original_blocks > CM256_MAX_BLOCKS) {
//   std::cerr << "CM256: Number of original blocks must be between " << CM256_MIN_BLOCKS << " and " << CM256_MAX_BLOCKS << " (is " << config_.computed.original_blocks << ").\n";
//   return -1;
// }

// if (config_.computed.recovery_blocks > CM256_MAX_BLOCKS-config_.computed.original_blocks) {
//   std::cerr << "CM256: Recovery blocks must be between 0 and " << CM256_MAX_BLOCKS-config_.computed.original_blocks << " (is " << config_.computed.recovery_blocks << ").\n";
//   return -1;
// }





// TODO: make this a variable and implement correctly
#define WARMUP_ITERATIONS 5

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
  }; // class BenchmarkRunner

#endif // BENCHMARK_RUNNER_H