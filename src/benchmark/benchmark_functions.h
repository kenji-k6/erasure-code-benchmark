/**
 * @file benchmark_functions.h
 * @brief Declares benchmark functions for different EC algorithms.
 */

#ifndef BENCHMARK_FUNCTIONS_H
#define BENCHMARK_FUNCTIONS_H

#include <benchmark/benchmark.h>
#include <benchmark_config.h>

/// Benchmark function declarations
void BM_Baseline(benchmark::State& state, const BenchmarkConfig& config);

void BM_BaselineScalar(benchmark::State& state, const BenchmarkConfig& config);
void BM_BaselineAVX(benchmark::State& state, const BenchmarkConfig& config);
void BM_BaselineAVX2(benchmark::State& state, const BenchmarkConfig& config);
void BM_CM256(benchmark::State& state, const BenchmarkConfig& config);
void BM_ISAL(benchmark::State& state, const BenchmarkConfig& config);
void BM_Leopard(benchmark::State& state, const BenchmarkConfig& config);
void BM_Wirehair(benchmark::State& state, const BenchmarkConfig& config);

constexpr uint32_t NUM_FULL_BENCHMARKS = 7;

#endif // BENCHMARK_FUNCTIONS_H