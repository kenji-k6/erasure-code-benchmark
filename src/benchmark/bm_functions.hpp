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
void BM_Wirehair(benchmark::State& state, const BenchmarkConfig& config);

void BM_XOREC(benchmark::State& state, const BenchmarkConfig& config);
void BM_XOREC_UNIFIED_PTR(benchmark::State& state, const BenchmarkConfig& config);
void BM_XOREC_GPU_PTR(benchmark::State& state, const BenchmarkConfig& config);
void BM_XOREC_GPU_CMP(benchmark::State& state, const BenchmarkConfig& config);

/// Performance Benchmark functions
void BM_XOR_BLOCKS_SCALAR(benchmark::State& state, const BenchmarkConfig& config);
void BM_XOR_BLOCKS_AVX(benchmark::State& state, const BenchmarkConfig& config);
void BM_XOR_BLOCKS_AVX2(benchmark::State& state, const BenchmarkConfig& config);
void BM_XOR_BLOCKS_AVX512(benchmark::State& state, const BenchmarkConfig& config);
#endif // BM_FUNCTIONS_HPP