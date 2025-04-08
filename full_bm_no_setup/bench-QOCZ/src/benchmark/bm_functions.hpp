/**
 * @file bm_functions.h
 * @brief Declares benchmark functions for different EC algorithms.
 */


#ifndef BM_FUNCTIONS_HPP
#define BM_FUNCTIONS_HPP

#include <benchmark/benchmark.h>
#include "bm_config.hpp"

/// EC-Benchmark functions
void BM_CM256(benchmark::State& state, const BenchmarkConfig& config);
void BM_ISAL(benchmark::State& state, const BenchmarkConfig& config);
void BM_Leopard(benchmark::State& state, const BenchmarkConfig& config);
void BM_XOREC_AVX(benchmark::State& state, const BenchmarkConfig& config);
void BM_XOREC_AVX2(benchmark::State& state, const BenchmarkConfig& config);
void BM_XOREC_AVX512(benchmark::State& state, const BenchmarkConfig& config);

#endif // BM_FUNCTIONS_HPP