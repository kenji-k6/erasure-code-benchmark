#ifndef BENCHMARK_GENERIC_RUNNER_HPP
#define BENCHMARK_GENERIC_RUNNER_HPP

#include "abstract_benchmark.hpp"
#include "benchmark/benchmark.h"
#include <chrono>
#include <vector>
#include <cmath>


/**
 * @brief Runs a generic benchmark for a given EC Library Benchmark type
 * 
 * This function is templated to work with any class that implements
 * the `ECBenchmark` interface. It follows a standard benchmarking procedure:
 * 1. Pauses timing and sets up the benchmark environment
 * 2. Touches GPU memory if required
 * 3. Encodes data
 * 4. Simulates data loss (untimed)
 * 5. Touches GPU memory if required
 * 6. Decodes data
 * 7. Verifies the correctness of the decoded data (untimed)
 * 8. Cleans up the benchmark environment
 * 
 * If data corruption is detected after decoding, the benchmark run is skipped
 * with an error message
 * 
 * @attention Pausing and stopping in each iteration incurs some overhead, however it is rouglhy 200-300ns and therefore negligible
 * Manual Timing as specified in the Google Benchmark documentation does not change this, since then the timings lose accuracy
 * 
 * @tparam BenchmarkType A class implementing the `ECBenchmark` interface
 * @param state The Google Benchmark state object
 */
template <typename BenchmarkType>
static void BM_generic(benchmark::State& state, const BenchmarkConfig& config) {
  std::vector<int64_t> enc_times(config.num_iterations);
  std::vector<double> enc_throughputs(config.num_iterations);

  std::vector<int64_t> dec_times(config.num_iterations);
  std::vector<double> dec_throughputs(config.num_iterations);

  double enc_time_mean = 0;
  double dec_time_mean = 0;

  double enc_throughput_mean = 0;
  double dec_throughput_mean = 0;

  unsigned it = 0;

  for (auto _ : state) {
    BenchmarkType bench(config);
    if (config.is_xorec_config && config.xorec_params.gpu_mem && config.xorec_params.touch_gpu_mem) bench.touch_gpu_memory();

    auto start_encode = std::chrono::steady_clock::now();
    bench.encode();
    auto end_encode = std::chrono::steady_clock::now();

    bench.simulate_data_loss();
    if (config.is_xorec_config && config.xorec_params.gpu_mem && config.xorec_params.touch_gpu_mem) bench.touch_gpu_memory();
    
    auto start_decode = std::chrono::steady_clock::now();
    bench.decode();
    auto end_decode = std::chrono::steady_clock::now();

    if (!bench.check_for_corruption()) {
      state.SkipWithMessage("Corruption Detected");
    }
    
    double time_encode = std::chrono::duration_cast<std::chrono::nanoseconds>(end_encode - start_encode).count();
    double time_decode = std::chrono::duration_cast<std::chrono::nanoseconds>(end_decode - start_decode).count();

    enc_time_mean += time_encode;
    dec_time_mean += time_decode;

    double enc_throughput = static_cast<double>((config.data_size * 8) / 1e9) / (time_encode / 1e9);
    double dec_throughput = static_cast<double>((config.data_size * 8) / 1e9) / (time_decode / 1e9);

    enc_throughput_mean += enc_throughput;
    dec_throughput_mean += dec_throughput;

    enc_times[it] = time_encode;
    dec_times[it] = time_decode;

    enc_throughputs[it] = enc_throughput;
    dec_throughputs[it] = dec_throughput;

    ++it;
    if (config.progress_reporter != nullptr) config.progress_reporter->update_bar();
    state.SetIterationTime(static_cast<double>(time_encode+time_decode)/1e9);
  }

  
  enc_time_mean /= config.num_iterations;
  dec_time_mean /= config.num_iterations;
  enc_throughput_mean /= config.num_iterations;
  dec_throughput_mean /= config.num_iterations;


  double enc_time_stddev = 0;
  double dec_time_stddev = 0;
  double enc_throughput_stddev = 0;
  double dec_throughput_stddev = 0;

  

  for (int i = 0; i < config.num_iterations; ++i) {
    enc_time_stddev += std::pow(enc_times[i] - enc_time_mean, 2);
    dec_time_stddev += std::pow(dec_times[i] - dec_time_mean, 2);

    enc_throughput_stddev += std::pow(enc_throughputs[i] - enc_throughput_mean, 2);
    dec_throughput_stddev += std::pow(dec_throughputs[i] - dec_throughput_mean, 2);
  }

  enc_time_stddev = std::sqrt(enc_time_stddev / (config.num_iterations-1));
  dec_time_stddev = std::sqrt(dec_time_stddev / (config.num_iterations-1));

  enc_throughput_stddev = std::sqrt(enc_throughput_stddev / (config.num_iterations-1));
  dec_throughput_stddev = std::sqrt(dec_throughput_stddev / (config.num_iterations-1));

  double enc_throughput = static_cast<double>((config.data_size * 8) / 1e9) / (enc_time_mean / 1e9);
  double dec_throughput = static_cast<double>((config.data_size * 8) / 1e9) / (dec_time_mean / 1e9);


  // Save results to counters
  state.counters["plot_id"] = config.plot_id;
  state.counters["tot_data_size_B"] = config.data_size;
  state.counters["block_size_B"] = config.block_size;
  state.counters["num_lost_blocks"] = config.num_lost_blocks;
  state.counters["redundancy_ratio"] = config.redundancy_ratio;
  state.counters["num_data_blocks"] = config.computed.num_original_blocks;
  state.counters["num_parity_blocks"] = config.computed.num_recovery_blocks;

  state.counters["encode_time_ns"] = enc_time_mean;
  state.counters["encode_time_ns_stddev"] = enc_time_stddev;
  state.counters["encode_throughput_Gbps"] = enc_throughput;
  state.counters["encode_throughput_Gbps_stddev"] = enc_throughput_stddev;

  state.counters["decode_time_ns"] = dec_time_mean;
  state.counters["decode_time_ns_stddev"] = dec_time_stddev;
  state.counters["decode_throughput_Gbps"] = dec_throughput;
  state.counters["decode_throughput_Gbps_stddev"] = dec_throughput_stddev;
}

#endif // BENCHMARK_GENERIC_RUNNER_HPP