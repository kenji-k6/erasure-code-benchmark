#include "benchmark_runner.h"
#include <iostream>


int main() {
  BenchmarkRunner runner;

  // Define a test case
  BenchmarkConfig config;
  config.data_size = 108800000; //1073736320; // ~~1.0737 GB
  config.block_size = 640000; //6'316'096; // 6316.096 KB
  config.redundancy_ratio = 0.5;
  config.loss_rate = 0.0;
  config.iterations = 4;

  //original count should be 170, recovery should be 85

  // Add the test case to the runner
  runner.add_test_case(BenchmarkRunner::Library::cm256, config);
  runner.add_test_case(BenchmarkRunner::Library::leopard, config);

  runner.run_all();

  return 0;
}