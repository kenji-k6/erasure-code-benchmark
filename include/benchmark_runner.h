#ifndef BENCHMARK_RUNNER_H
#define BENCHMARK_RUNNER_H

#include "abstract_benchmark.h"
#include "benchmark/benchmark.h"


/**
 * @brief Runs a generic benchmark for a given ECC Library Benchmark type
 * 
 * This function is templated to work with any class that implements
 * the `ECCBenchmark` interface. It follows a standard benchmarking procedure:
 * 1. Pauses timing and sets up the benchmark environment
 * 2. Encodes data
 * 3. Simulates data loss (untimed)
 * 4. Decodes data
 * 5. Verifies the correctness of the decoded data (untimed)
 * 6. Cleans up the benchmark environment
 * 
 * If data corruption is detected after decoding, the benchmark run is skipped
 * with an error message
 * 
 * @tparam BenchmarkType A class implementing the `ECCBenchmark` interface
 * @param state The Google Benchmark state object
 */
template <typename BenchmarkType>
static void BM_generic(benchmark::State& state) {
  BenchmarkType bench;
  for (auto _ : state) {
    state.PauseTiming();
    bench.setup();
    state.ResumeTiming();

    bench.encode();
    state.PauseTiming();
    bench.simulate_data_loss();
    // bench.flush_cache(); // Uncomment if cache flushing is required
    state.ResumeTiming();
    bench.decode();

    state.PauseTiming();
    if (!bench.check_for_corruption()) {
      state.SkipWithError("Corruption detected");
    }
    bench.teardown();
    state.ResumeTiming();
  }
}

#endif // BENCHMARK_RUNNER_H