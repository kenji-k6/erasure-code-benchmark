/**
 * @file runners.hpp
 * @brief Declares benchmark functions for different EC algorithms.
 */


#ifndef RUNNERS_HPP
#define RUNNERS_HPP

#include <benchmark/benchmark.h>
#include "bm_config.hpp"

/// EC-Benchmark functions
void BM_CM256(benchmark::State& state, const BenchmarkConfig& config);
void BM_ISAL(benchmark::State& state, const BenchmarkConfig& config);
void BM_Leopard(benchmark::State& state, const BenchmarkConfig& config);

void BM_XOREC(benchmark::State& state, const BenchmarkConfig& config);
void BM_XOREC_UNIFIED_PTR(benchmark::State& state, const BenchmarkConfig& config);
void BM_XOREC_GPU_PTR(benchmark::State& state, const BenchmarkConfig& config);
void BM_XOREC_GPU_CMP(benchmark::State& state, const BenchmarkConfig& config);
#endif // RUNNERS_HPP