#include "benchmark_runner.h"
#include "leopard_benchmark.h"
#include "benchmark/benchmark.h"
#include "leopard.h"
#include <memory>
#include <iostream>
#include <functional>


// TODO: Pass arguments
BenchmarkConfig get_config() {
  BenchmarkConfig config;
  config.data_size = 10880000; //1073736320; // ~~1.0737 GB
  config.block_size = 64000; //6'316'096; // 6316.096 KB
  config.redundancy_ratio = 0.5;
  config.loss_rate = 0.0;
  config.iterations = 4;
  config.computed.original_blocks = (config.data_size + (config.block_size - 1)) / config.block_size;
  return config;
}

int main(int argc, char** argv) {

  BenchmarkConfig config = get_config();

  // Register Benchmarks


  // Default argument if no arguments are passed
  char arg0_default[] = "benchmark";  
  char* args_default = arg0_default;

  // If no arguments are passed, set argc to 1 and argv to point to the default argument
  if (argc == 0 || argv == nullptr) {
    argc = 1;
    argv = &args_default;
  }

  // Initialize Google Benchmark
  ::benchmark::Initialize(&argc, argv);

  // Check and report unrecognized arguments
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
      return 1; // Return error code if there are unrecognized arguments
  }


  // Run all specified benchmarks
  ::benchmark::RunSpecifiedBenchmarks();

  // Shutdown Google Benchmark
  ::benchmark::Shutdown();


}