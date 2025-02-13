#include "benchmark_runner.h"


/*
 *
*/

// Add a test case to the runner
void BenchmarkRunner::add_test_case(Library lib, const BenchmarkConfig& config) {
  BenchmarkConfig computed_config = config;
  compute_config(computed_config);
  test_cases_.emplace_back(lib, computed_config);
}


// Run all added test cases
void BenchmarkRunner::run_all() {
  for (const auto& [lib, config] : test_cases_) {
    std::unique_ptr<ECCBenchmark> bench = create_benchmark(lib);
    if (!bench) {
      std::cerr << "Failed to create benchmark for library " << static_cast<int>(lib) << ".\n";
      continue;
    }

    if (!bench->setup(config)) {
      std::cerr << "Failed to setup benchmark for library " << static_cast<int>(lib) << ".\n";
      continue;
    }

    run_single(*bench, config);
    bench->teardown();

    save_results(lib, config, bench->get_metrics());
  }
}


// Create a benchmark instance for a specific library
std::unique_ptr<ECCBenchmark> BenchmarkRunner::create_benchmark(Library lib) {
  switch (lib) {
    case Library::aff3ct:
      // TODO
      throw std::runtime_error("Wirehair benchmark not implemented.");
    
    case Library::cm256:
      // TODO
      throw std::runtime_error("cm256 benchmark not implemented.");
    
    case Library::isa_l:
      // TODO
      throw std::runtime_error("ISA-L benchmark not implemented.");
    
    case Library::leopard:
      return std::make_unique<LeopardBenchmark>();
    
    case Library::wirehair:
      // TODO
      throw std::runtime_error("Wirehair benchmark not implemented.");
  }
}

