#ifndef ABSTRACT_RUNNER_HPP
#define ABSTRACT_RUNNER_HPP

#include "abstract_bm.hpp"
#include "benchmark/benchmark.h"
#include <chrono>
#include <vector>
#include <cmath>
#include <numeric>


/**
 * @brief Runs a generic benchmark for a given EC Library Benchmark type
 * 
 * This function is templated to work with any class that implements
 * the `AbstractBenchmark` interface. It follows a standard benchmarking procedure:
 * 1. Pauses timing and sets up the benchmark environment
 * 2. Touches unified memory if required
 * 3. Encodes data
 * 4. Simulates data loss (untimed)
 * 5. Touches unified memory if required
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
 * @tparam BenchmarkType A class implementing the `AbstractBenchmark` interface
 * @param state The Google Benchmark state object
 */
template <typename BenchmarkType>
static void BM_generic(benchmark::State& state, const BenchmarkConfig& config) {
  std::vector<double> enc_times(config.num_iterations);
  std::vector<double> dec_times(config.num_iterations);

  unsigned it = 0;

  BenchmarkType bench(config);
  for (int i = 0; i < config.num_warmup_iterations; ++i) {
    bench.setup();
    bench.encode();
    bench.simulate_data_loss();
    bench.decode();
  }
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
    
    double time_encode = std::chrono::duration_cast<std::chrono::nanoseconds>(end_encode - start_encode).count();
    double time_decode = std::chrono::duration_cast<std::chrono::nanoseconds>(end_decode - start_decode).count();

    enc_times[it] = time_encode;
    dec_times[it++] = time_decode;

    if (config.reporter != nullptr) config.reporter->update_bar();
    state.SetIterationTime(static_cast<double>(time_encode+time_decode)/1e9);
  }


  double enc_sum = std::accumulate(enc_times.begin(), enc_times.end(), 0.0);
  double dec_sum = std::accumulate(dec_times.begin(), dec_times.end(), 0.0);
  

  double enc_t_ns = enc_sum / config.num_iterations;
  double dec_t_ns = dec_sum / config.num_iterations;

  double enc_accum = 0.0;
  double dec_accum = 0.0;

  std::for_each(enc_times.begin(), enc_times.end(), [&](const double t) {
    enc_accum += std::pow(t - enc_t_ns, 2);
  });

  std::for_each(dec_times.begin(), dec_times.end(), [&](const double t) {
    dec_accum += std::pow(t - dec_t_ns, 2);
  });

  double enc_t_ns_stddev = std::sqrt(enc_accum / (config.num_iterations-1));
  double dec_t_ns_stddev = std::sqrt(dec_accum / (config.num_iterations-1));

  // equal to (#bits / 10^9) / (t_ns / 10^9) = #Gbits / s
  double enc_tp_Gbps = (config.data_size * 8) / enc_t_ns;
  double dec_tp_Gbps = (config.data_size * 8) / dec_t_ns;

  // First order talor approximation for stddev of throughput
  double enc_tp_Gbps_stddev = enc_tp_Gbps * (enc_t_ns_stddev / enc_t_ns);
  double dec_tp_Gbps_stddev = dec_tp_Gbps * (dec_t_ns_stddev / dec_t_ns);


  // Save results to counters
  state.counters["num_warmup_iterations"] = config.num_warmup_iterations;

  state.counters["data_size_B"] = config.data_size;
  state.counters["block_size_B"] = config.block_size;
  state.counters["ec_params_0"] = get<0>(config.ec_params);
  state.counters["ec_params_1"] = get<1>(config.ec_params);
  state.counters["num_lost_blocks"] = config.num_lost_blocks;


  state.counters["gpu_computation"] = (config.gpu_computation) ? 1 : 0;
  state.counters["num_gpu_blocks"] = config.num_gpu_blocks;
  state.counters["threads_per_gpu_block"] = config.threads_per_gpu_block;

  state.counters["encode_time_ns"] = enc_t_ns;
  state.counters["encode_time_ns_stddev"] = enc_t_ns_stddev;
  state.counters["encode_throughput_Gbps"] = enc_tp_Gbps;
  state.counters["encode_throughput_Gbps_stddev"] = enc_tp_Gbps_stddev;

  state.counters["decode_time_ns"] = dec_t_ns;
  state.counters["decode_time_ns_stddev"] = dec_t_ns_stddev;
  state.counters["decode_throughput_Gbps"] = dec_tp_Gbps;
  state.counters["decode_throughput_Gbps_stddev"] = dec_tp_Gbps_stddev;
}

#endif // ABSTRACT_RUNNER_HPP