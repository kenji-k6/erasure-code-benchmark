/**
 * @file benchmark_functions.h
 * @brief Declares benchmark functions for different EC algorithms.
 */

#ifndef BENCHMARK_FUNCTIONS_H
#define BENCHMARK_FUNCTIONS_H

#include <benchmark/benchmark.h>
#include <benchmark_config.h>

/// Benchmark function declarations
void BM_XOREC(benchmark::State& state, const BenchmarkConfig& config);
void BM_XOREC_Scalar(benchmark::State& state, const BenchmarkConfig& config);
void BM_XOREC_AVX(benchmark::State& state, const BenchmarkConfig& config);
void BM_XOREC_AVX2(benchmark::State& state, const BenchmarkConfig& config);
void BM_CM256(benchmark::State& state, const BenchmarkConfig& config);
void BM_ISAL(benchmark::State& state, const BenchmarkConfig& config);
void BM_Leopard(benchmark::State& state, const BenchmarkConfig& config);
void BM_Wirehair(benchmark::State& state, const BenchmarkConfig& config);



void BM_CUDA_XOREC(benchmark::State& state, const BenchmarkConfig& config);
#endif // BENCHMARK_FUNCTIONS_H