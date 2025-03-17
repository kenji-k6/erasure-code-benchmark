/**
 * @file benchmark_functions.h
 * @brief Declares benchmark functions for different EC algorithms.
 */


#ifndef BENCHMARK_FUNCTIONS_HPP
#define BENCHMARK_FUNCTIONS_HPP

#include <benchmark/benchmark.h>
#include "benchmark_config.hpp"

/// Benchmark function declarations
void BM_CM256(benchmark::State& state, const BenchmarkConfig& config);
void BM_ISAL(benchmark::State& state, const BenchmarkConfig& config);
void BM_Leopard(benchmark::State& state, const BenchmarkConfig& config);
void BM_Wirehair(benchmark::State& state, const BenchmarkConfig& config);

void BM_XOREC(benchmark::State& state, const BenchmarkConfig& config);
void BM_XOREC_UNIFIED_PTR(benchmark::State& state, const BenchmarkConfig& config);
void BM_XOREC_GPU_PTR(benchmark::State& state, const BenchmarkConfig& config);
void BM_XOREC_GPU_CMP(benchmark::State& state, const BenchmarkConfig& config);
#endif // BENCHMARK_FUNCTIONS_HPP