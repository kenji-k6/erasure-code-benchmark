#ifndef BM_GENERIC_RUNNER_HPP
#define BM_GENERIC_RUNNER_HPP

#include "abstract_bm.hpp"
#include "benchmark/benchmark.h"
#include <chrono>
#include <vector>
#include <cmath>


template <typename BenchmarkType>
static void BM_generic(benchmark::State& state, const BenchmarkConfig& config) {
  BenchmarkType bench(config);
  for (auto _ : state) {
    bench.encode();

    state.PauseTiming();

    bench.simulate_data_loss();
    bench.decode();

    if (!bench.check_for_corruption()) {
      state.counters["data_corrupted"] = 1;
    }
    state.ResumeTiming();
  }

  if (config.progress_reporter != nullptr) {
    config.progress_reporter->update_bar();
  }

  state.counters["message_size_B"] = config.message_size;
  state.counters["block_size_B"] = config.block_size;
  state.counters["fec_params_0"] = get<0>(config.fec_params);
  state.counters["fec_params_1"] = get<1>(config.fec_params);
  state.counters["num_lost_rmda_packets"] = config.num_lost_rmda_packets;

  state.counters["is_gpu_bm"] = config.is_gpu_config ? 1 : 0;
  state.counters["num_gpu_blocks"] = config.num_gpu_blocks;
  state.counters["threads_per_gpu_block"] = config.threads_per_gpu_block;
}

#endif // BM_GENERIC_RUNNER_HPP