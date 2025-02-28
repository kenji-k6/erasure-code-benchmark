#ifndef GENERIC_BENCHMARK_RUNNER_H
#define GENERIC_BENCHMARK_RUNNER_H

#include "abstract_benchmark.h"
#include "benchmark/benchmark.h"
#include <chrono>
#include <vector>
#include <cmath>


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
 * @attention Pausing and stopping in each iteration incurs some overhead, however it is rouglhy 200-300ns and therefore negligible
 * Manual Timing as specified in the Google Benchmark documentation does not change this, since then the timings lose accuracy
 * 
 * @tparam BenchmarkType A class implementing the `ECCBenchmark` interface
 * @param state The Google Benchmark state object
 */
template <typename BenchmarkType>
static void BM_generic(benchmark::State& state, const BenchmarkConfig& config) {
  BenchmarkType bench(config);
  std::vector<int64_t> enc_times(config.num_iterations);
  std::vector<int64_t> dec_times(config.num_iterations);
  double enc_mean = 0;
  double dec_mean = 0;
  unsigned it = 0;

  for (auto _ : state) {
    bench.setup();

    auto start_encode = std::chrono::steady_clock::now();
    bench.encode();
    auto end_encode = std::chrono::steady_clock::now();

    bench.simulate_data_loss();
    
    auto start_decode = std::chrono::steady_clock::now();
    bench.decode();
    auto end_decode = std::chrono::steady_clock::now();

    if (!bench.check_for_corruption()) {
      state.SkipWithMessage("Corruption Detected");
    }
    bench.teardown();


    int64_t time_encode = std::chrono::duration_cast<std::chrono::nanoseconds>(end_encode - start_encode).count();
    int64_t time_decode = std::chrono::duration_cast<std::chrono::nanoseconds>(end_decode - start_decode).count();

    enc_mean += time_encode;
    dec_mean += time_decode;

    enc_times[it] = time_encode;
    dec_times[it] = time_decode;
    ++it;
    
    state.SetIterationTime(static_cast<double>(time_encode+time_decode)/1e9);
  }

  // Compute the standard deviation of the encode, decode and total times aswell as the max and min times for encoding, decoding and total
  enc_mean /= config.num_iterations;
  dec_mean /= config.num_iterations;
  double tot_mean = enc_mean + dec_mean;
  double enc_stddev = 0;
  double dec_stddev = 0;
  double tot_stddev = 0;

  long enc_max = enc_times[0];
  long dec_max = dec_times[0];
  long tot_max = enc_times[0] + dec_times[0];

  long enc_min = enc_times[0];
  long dec_min = dec_times[0];
  long tot_min = enc_times[0] + dec_times[0];
  for (unsigned i = 0; i < config.num_iterations; ++i) {
    enc_stddev += std::pow(enc_times[i] - enc_mean, 2);
    dec_stddev += std::pow(dec_times[i] - dec_mean, 2);
    tot_stddev += std::pow((enc_times[i] + dec_times[i]) - tot_mean, 2);

    enc_max = std::max(enc_max, enc_times[i]);
    dec_max = std::max(dec_max, dec_times[i]);
    tot_max = std::max(tot_max, enc_times[i] + dec_times[i]);

    enc_min = std::min(enc_min, enc_times[i]);
    dec_min = std::min(dec_min, dec_times[i]);
    tot_min = std::min(tot_min, enc_times[i] + dec_times[i]);
  }

  enc_stddev = std::sqrt(enc_stddev / (config.num_iterations-1));
  dec_stddev = std::sqrt(dec_stddev / (config.num_iterations-1));
  tot_stddev = std::sqrt(tot_stddev / (config.num_iterations-1));


  // Save results to counters
  state.counters["encode_time_ns"] = enc_mean;
  state.counters["decode_time_ns"] = dec_mean;
  state.counters["encode_time_stddev_ns"] = enc_stddev;
  state.counters["decode_time_stddev_ns"] = dec_stddev;
  state.counters["tot_time_stddev_ns"] = tot_stddev;
  state.counters["encode_time_max_ns"] = enc_max;
  state.counters["decode_time_max_ns"] = dec_max;
  state.counters["tot_time_max_ns"] = tot_max;
  state.counters["encode_time_min_ns"] = enc_min;
  state.counters["decode_time_min_ns"] = dec_min;
  state.counters["tot_time_min_ns"] = tot_min;

  state.counters["tot_data_size_B"] = config.data_size;
  state.counters["block_size_B"] = config.block_size;
  state.counters["num_lost_blocks"] = config.num_lost_blocks;
  state.counters["redundancy_ratio"] = config.redundancy_ratio;
  state.counters["plot_id"] = config.plot_id;

}

#endif // GENERIC_BENCHMARK_RUNNER_H