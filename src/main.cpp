#include "benchmark_runner.h"
#include "leopard_benchmark.h"
#include "cm256_benchmark.h"
#include "wirehair_benchmark.h"
#include "isal_benchmark.h"
#include "baseline_benchmark.h"
#include "benchmark/benchmark.h"
#include "utils.h"

#include <memory>
#include <iostream>
#include <cmath>

BenchmarkConfig benchmark_config;
std::vector<uint32_t> lost_block_idxs = {};



//TODO: check if input config is valid and for which libraries it is valid
//TODO: check that lost_blocks <= recovery_blocks

// TODO: Pass arguments
BenchmarkConfig get_config() {
  BenchmarkConfig config;
  config.data_size = 81'280'000; //1073736320; // ~~1.0737 GB
  config.block_size = 640000; //6'316'096; // 6316.096 KB
  config.num_lost_blocks = 10;
  config.redundancy_ratio = 1;
  config.num_iterations = 1;
  config.computed.num_original_blocks = (config.data_size + (config.block_size - 1)) / config.block_size;
  config.computed.num_recovery_blocks = static_cast<size_t>(std::ceil(config.computed.num_original_blocks * config.redundancy_ratio));
  return config;
}

static void BM_cm256(benchmark::State& state) {
  BM_generic<CM256Benchmark>(state);
}

static void BM_leopard(benchmark::State& state) {
  BM_generic<LeopardBenchmark>(state);
}

/*
 * Important: Wirehair does not accept a specified no. of recovery blocks
 * It keeps sending blocks until the decoder has enough to recover the original data
*/
static void BM_wirehair(benchmark::State& state) {
  BM_generic<WirehairBenchmark>(state);
}

static void BM_isal(benchmark::State& state) {
  BM_generic<ISALBenchmark>(state);
}

static void BM_baseline(benchmark::State& state) {
  BM_generic<BaselineBenchmark>(state);
}



int main(int argc, char** argv) {

  // Get and compute the configuration
  benchmark_config = get_config();

  // Get the lost block indexes
  select_lost_block_idxs(
    benchmark_config.num_lost_blocks,
    benchmark_config.computed.num_original_blocks + benchmark_config.computed.num_recovery_blocks,
    lost_block_idxs
  );

  // Register Benchmarks
  BENCHMARK(BM_cm256)->Iterations(benchmark_config.num_iterations);
  BENCHMARK(BM_leopard)->Iterations(benchmark_config.num_iterations);
  BENCHMARK(BM_wirehair)->Iterations(benchmark_config.num_iterations);
  BENCHMARK(BM_isal)->Iterations(benchmark_config.num_iterations);
  BENCHMARK(BM_baseline)->Iterations(benchmark_config.num_iterations);


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