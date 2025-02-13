#ifndef BENCHMARK_RUNNER_H
#define BENCHMARK_RUNNER_H

#include "benchmark.h"
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