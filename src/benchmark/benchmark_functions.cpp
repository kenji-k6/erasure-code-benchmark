/**
 * @file benchmark_functions.cpp
 * @brief Implements benchmark functions for different EC algorithms.
 */

#include "benchmark_functions.h"
#include "benchmark_generic_runner.h"
#include "xorec_benchmark.h"
#include "cm256_benchmark.h"
#include "isal_benchmark.h"
#include "leopard_benchmark.h"
#include "wirehair_benchmark.h"


void BM_XOREC(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XORECBenchmark>(state, config);
}

void BM_XOREC_Scalar(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XORECScalarBenchmark>(state, config);
}

void BM_XOREC_AVX(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XORECAVXBenchmark>(state, config);
}

void BM_XOREC_AVX2(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XORECAVX2Benchmark>(state, config);
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


void BM_XOREC_GPU(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XORECBenchmarkGPU>(state, config);
}

void BM_XOREC_Scalar_GPU(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XORECScalarBenchmarkGPU>(state, config);
}

void BM_XOREC_AVX_GPU(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XORECAVXBenchmarkGPU>(state, config);
}

void BM_XOREC_AVX2_GPU(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XORECAVX2BenchmarkGPU>(state, config);
}