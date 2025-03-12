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


void BM_XOREC(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XorecBenchmark>(state, config);
}

void BM_XOREC_Scalar(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XorecBenchmarkScalar>(state, config);
}

void BM_XOREC_AVX(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XorecBenchmarkAVX>(state, config);
}

void BM_XOREC_AVX2(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XorecBenchmarkAVX2>(state, config);
}


void BM_XOREC_GPU_POINTER(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XorecBenchmarkGPUPointer>(state, config);
}

void BM_XOREC_Scalar_GPU_POINTER(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XorecBenchmarkScalarGPUPointer>(state, config);
}

void BM_XOREC_AVX_GPU_POINTER(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XorecBenchmarkAVXGPUPointer>(state, config);
}

void BM_XOREC_AVX2_GPU_POINTER(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XorecBenchmarkAVX2GPUPointer>(state, config);
}


void BM_XOREC_GPU_COMPUTATION(benchmark::State&state, const BenchmarkConfig& config) {
  BM_generic<XorecBenchmarkGPUComputation>(state, config);
}