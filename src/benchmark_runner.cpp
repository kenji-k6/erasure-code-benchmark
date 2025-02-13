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
  for (const auto& [lib, computed_config] : test_cases_) {
    std::unique_ptr<ECCBenchmark> bench = create_benchmark(lib);
    if (!bench) {
      std::cerr << "Failed to create benchmark for library " << static_cast<int>(lib) << ".\n";
      continue;
    }

    if (bench->setup(computed_config)) {
      std::cerr << "Failed to setup benchmark for library " << static_cast<int>(lib) << ".\n";
      continue;
    }

    run_single(*bench, computed_config);
    bench->teardown();

    save_results(lib, computed_config, bench->get_metrics());
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


// TODO: properly take measurements
void BenchmarkRunner::run_single(ECCBenchmark& bench, const BenchmarkConfig& computed_config) {
  for (int i = 0; i < WARMUP_ITERATIONS; i++) {
    bench.encode();
    bench.decode(computed_config.loss_rate);
  }

  for (int i = 0; i < computed_config.iterations; i++) {
    if (bench.encode()) {
      std::cerr << "Encode failed for iteration " << i << ".\n";
    }

    if (bench.decode(computed_config.loss_rate)) {
      std::cerr << "Decode failed for iteration " << i << ".\n";
    }

    ECCBenchmark::Metrics metrics = bench.get_metrics();

    // total data size in MiB (power of 2)
    double total_mib = (computed_config.computed.original_blocks * computed_config.block_size) / MEGABYTE_TO_BYTE_FACTOR;

    std::cout << "Iteration " << i << ":\n"
              << "Encoder(" << total_mib << " MiB in "
              << computed_config.computed.original_blocks << " blocks): Input="
              << metrics.encode_input_throughput_mbps << " Mbps, Output="
              << metrics.encode_output_throughput_mbps << " Mbps\n";
    
    std::cout << "Decoder(" << total_mib << " MiB in "
              << computed_config.computed.original_blocks << " blocks): Input="
              << metrics.decode_input_throughput_mbps << " Mbps, Output="
              << metrics.decode_output_throughput_mbps << " Mbps\n";
  }
}


void BenchmarkRunner::save_results(Library lib, const BenchmarkConfig& config, const ECCBenchmark::Metrics& metrics) {
  // TODO: implement
  return;
}


void BenchmarkRunner::compute_config(BenchmarkConfig& config) {
  config.computed.original_blocks = (config.data_size + (config.block_size - 1)) / config.block_size;
  config.computed.recovery_blocks = static_cast<size_t>(std::ceil(config.computed.original_blocks * config.redudandy_ratio));
}