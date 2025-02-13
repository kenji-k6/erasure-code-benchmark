#include "benchmark_runner.h"
#include <iostream>


int main() {
  BenchmarkRunner runner;

  // Define a test case
  BenchmarkConfig config;
  config.data_size = 1'073'741'824; // 8 GiB
  config.block_size = 131'072; // 128 KiB
  config.redundancy_ratio = 0.4;
  config.loss_rate = 0.0;
  config.iterations = 10;

  // Add the test case to the runner
  runner.add_test_case(BenchmarkRunner::Library::leopard, config);

  runner.run_all();

  return 0;
}