/**
 * @file benchmark_functions.cpp
 * @brief Implements benchmark functions for different EC algorithms.
 */

#include "benchmark_functions.h"
#include "benchmark_generic_runner.h"
#include "baseline_benchmark.h"
#include "cm256_benchmark.h"
#include "isal_benchmark.h"
#include "leopard_benchmark.h"
#include "wirehair_benchmark.h"

void BM_Baseline(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<BaselineBenchmark>(state, config);
}

void BM_BaselineScalar(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<BaselineScalarBenchmark>(state, config);
}

void BM_BaselineAVX(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<BaselineAVXBenchmark>(state, config);
}

void BM_BaselineAVX2(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<BaselineAVX2Benchmark>(state, config);
}

void BM_CM256(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<CM256Benchmark>(state, config);
}

void BM_ISAL(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<ISALBenchmark>(state, config);
}

void BM_Leopard(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<LeopardBenchmark>(state, config);
}

void BM_Wirehair(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<WirehairBenchmark>(state, config);
}