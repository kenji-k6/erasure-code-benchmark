#ifndef BM_GENERIC_RUNNER_HPP
#define BM_GENERIC_RUNNER_HPP

#include "abstract_bm.hpp"
#include "benchmark/benchmark.h"
#include <chrono>
#include <vector>
#include <cmath>


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
  state.counters["message_size_B"] = config.message_size;
  state.counters["block_size_B"] = config.block_size;
  state.counters["fec_params_0"] = get<0>(config.fec_params);
  state.counters["fec_params_1"] = get<1>(config.fec_params);
  state.counters["num_lost_rmda_packets"] = config.num_lost_rmda_packets;

  state.counters["is_gpu_bm"] = config.is_gpu_config ? 1 : 0;
  state.counters["num_gpu_blocks"] = config.num_gpu_blocks;
  state.counters["threads_per_gpu_block"] = config.threads_per_gpu_block;

  state.counters["encode_time_ns"] = enc_time_mean;
  state.counters["encode_time_ns_stddev"] = enc_time_stddev;
  state.counters["encode_throughput_Gbps"] = enc_throughput;
  state.counters["encode_throughput_Gbps_stddev"] = enc_throughput_stddev;

  state.counters["decode_time_ns"] = dec_time_mean;
  state.counters["decode_time_ns_stddev"] = dec_time_stddev;
  state.counters["decode_throughput_Gbps"] = dec_throughput;
  state.counters["decode_throughput_Gbps_stddev"] = dec_throughput_stddev;
}

#endif // BM_GENERIC_RUNNER_HPP