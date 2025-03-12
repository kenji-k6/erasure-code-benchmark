/**
 * @file benchmark_functions.cpp
 * @brief Implements benchmark functions for different EC algorithms.
 */

#include "benchmark_functions.h"
#include "benchmark_generic_runner.h"
#include "cm256_benchmark.h"
#include "isal_benchmark.h"
#include "leopard_benchmark.h"
#include "wirehair_benchmark.h"
#include "xorec_benchmark.h"
#include "xorec_gpu_ptr_benchmark.h"
#include "xorec_gpu_cmp_benchmark.h"



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


void BM_XOREC_SCALAR(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XorecBenchmarkScalar>(state, config);
}

void BM_XOREC_AVX(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XorecBenchmarkAVX>(state, config);
}

void BM_XOREC_AVX2(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XorecBenchmarkAVX2>(state, config);
}


void BM_XOREC_SCALAR_GPU_POINTER(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XorecBenchmarkScalarGPUPtr>(state, config);
}

void BM_XOREC_AVX_GPU_POINTER(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XorecBenchmarkAVXGPUPtr>(state, config);
}

void BM_XOREC_AVX2_GPU_POINTER(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XorecBenchmarkAVX2GPUPtr>(state, config);
}


void BM_XOREC_GPU_COMPUTATION(benchmark::State&state, const BenchmarkConfig& config) {
  BM_generic<XorecBenchmarkGPUCmp>(state, config);
}