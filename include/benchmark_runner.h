#ifndef BENCHMARK_RUNNER_H
#define BENCHMARK_RUNNER_H

#include "abstract_benchmark.h"
#include "utils.h"
#include "benchmark/benchmark.h"

//TODO: comments below

// // Assert that the block size is a multiple of 64 bytes
// if (config_.block_size % LEOPARD_BLOCK_SIZE_ALIGNMENT != 0) {
//   std::cerr << "Leopard: Block size must be a multiple of " << LEOPARD_BLOCK_SIZE_ALIGNMENT << " bytes.\n";
//   return -1;
// }

// // Assert that the number of blocks is within the valid range
// if (config_.computed.original_blocks < LEOPARD_MIN_BLOCKS || config_.computed.original_blocks > LEOPARD_MAX_BLOCKS) {
//   std::cerr << "Leopard: Original blocks must be between " << LEOPARD_MIN_BLOCKS << " and " << LEOPARD_MAX_BLOCKS << " (is " << config_.computed.original_blocks << ").\n";
//   return -1;
// }

// if (config_.computed.original_blocks < CM256_MIN_BLOCKS || config.computed.original_blocks > CM256_MAX_BLOCKS) {
//   std::cerr << "CM256: Number of original blocks must be between " << CM256_MIN_BLOCKS << " and " << CM256_MAX_BLOCKS << " (is " << config_.computed.original_blocks << ").\n";
//   return -1;
// }

// if (config_.computed.recovery_blocks > CM256_MAX_BLOCKS-config_.computed.original_blocks) {
//   std::cerr << "CM256: Recovery blocks must be between 0 and " << CM256_MAX_BLOCKS-config_.computed.original_blocks << " (is " << config_.computed.recovery_blocks << ").\n";
//   return -1;
// }

template <typename BenchmarkType>
static void BM_generic(benchmark::State& state) {
  BenchmarkType bench;
  bench.setup();
  for (auto _ : state) {
    bench.encode();

    state.PauseTiming();
    bench.simulate_data_loss();
    // bench.flush_cache();
    state.ResumeTiming();

    bench.decode();

    state.PauseTiming();
    if (bench.check_for_corruption()) {
      state.SkipWithError("Corruption detected");
    }
    state.ResumeTiming();
  }
}


// static void BM_cm256(benchmark::State& state);


// static void BM_leopard(benchmark::State& state);

#endif // BENCHMARK_RUNNER_H