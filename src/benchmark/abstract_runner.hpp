#ifndef ABSTRACT_RUNNER_HPP
#define ABSTRACT_RUNNER_HPP

#include "abstract_bm.hpp"
#include "benchmark/benchmark.h"
#include <chrono>
#include <vector>
#include <cmath>
#include <numeric>
#include <ranges>


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

  struct Stats {
    double t_mean;
    double t_stddev;
    double tp_mean;
    double tp_stddev;
  };

  auto run_warmup = [](BenchmarkType& bench, const BenchmarkConfig& cfg) {
    for (int i = 0; i < cfg.num_warmup_iterations; ++i) {
      bench.setup();
      bench.encode();
      bench.simulate_data_loss();
      bench.decode();
    }
  };

  auto compute_stats = [](const std::vector<double>& times, size_t data_size) {
    // Encode statistics (time is in nanoseconds)
    const double time_mean =  std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    const double time_accum = std::transform_reduce(
      times.begin(), times.end(), 0.0,
      std::plus<>(),
      [time_mean](double t) { return std::pow(t-time_mean, 2); }
    );
    const double time_stddev = std::sqrt(time_accum / (times.size()-1));
    
    // Throughput statistics (throughput is in Gbit/s)
    const size_t bits = data_size * 8;
    // the TP computation is equivalent to (#bits / 10^9) / (t_ns / 10^9) = #Gbits / s
    auto throughputs = std::views::transform([bits](double t) { return bits / t; })(times);
    const double throughput_mean = std::accumulate(throughputs.begin(), throughputs.end(), 0.0) / throughputs.size();
    const double throughput_accum = std::transform_reduce(
      throughputs.begin(), throughputs.end(), 0.0,
      std::plus<>(),
      [throughput_mean](double tp) { return std::pow(tp-throughput_mean, 2); }
    );
    const double throughput_stddev = std::sqrt(throughput_accum / (throughputs.size()-1));

    return Stats{
      .t_mean = time_mean,
      .t_stddev = time_stddev,
      .tp_mean = throughput_mean,
      .tp_stddev = throughput_stddev,
    };
  };

  auto set_counters = [&state](const std::string& prefix, const Stats& stats) {
    state.counters[prefix + "_time_ns"] = stats.t_mean;
    state.counters[prefix + "_time_ns_stddev"] = stats.t_stddev;
    state.counters[prefix + "_throughput_Gbps"] = stats.tp_mean;
    state.counters[prefix + "_throughput_Gbps_stddev"] = stats.tp_stddev;
  };


  std::vector<double> enc_times(config.num_iterations);
  std::vector<double> dec_times(config.num_iterations);
  unsigned it = 0;

  BenchmarkType bench(config);
  run_warmup(bench, config);

  for (auto _ : state) {
    //Timing loop
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

    if (config.reporter) config.reporter->update_bar();
    state.SetIterationTime(static_cast<double>(time_encode+time_decode)/1e9);
  }


  const Stats enc_stats = compute_stats(enc_times, config.data_size);
  const Stats dec_stats = compute_stats(dec_times, config.data_size);

  // Set counters
  const std::vector<std::pair<std::string, double>> config_counters = {
    { "num_warmup_iterations",  static_cast<double>(config.num_warmup_iterations)   },
    { "message_size_B",         static_cast<double>(config.message_size)            },
    { "block_size_B",           static_cast<double>(config.block_size)              },
    { "ec_params_0",            static_cast<double>(std::get<0>(config.ec_params))  },
    { "ec_params_1",            static_cast<double>(std::get<1>(config.ec_params))  },
    { "num_lost_blocks",        static_cast<double>(config.num_lost_blocks)         },
    { "num_cpu_threads",        static_cast<double>(config.num_cpu_threads)         },
    { "gpu_computation",        config.gpu_computation ? 1.0 : 0.0                  },
    { "num_gpu_blocks",         static_cast<double>(config.num_gpu_blocks)          },
    { "threads_per_gpu_block",  static_cast<double>(config.threads_per_gpu_block)   }
  };
  
  for (const auto& [key, value] : config_counters) {
    state.counters[key] = value;
  }
  set_counters("encode", enc_stats);
  set_counters("decode", dec_stats);
}

#endif // ABSTRACT_RUNNER_HPP