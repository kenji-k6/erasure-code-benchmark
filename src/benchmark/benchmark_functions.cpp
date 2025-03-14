/**
 * @file benchmark_functions.cpp
 * @brief Implements benchmark functions for different EC algorithms.
 */

#include "benchmark_functions.hpp"
#include "benchmark_generic_runner.hpp"
#include "cm256_benchmark.hpp"
#include "isal_benchmark.hpp"
#include "leopard_benchmark.hpp"
#include "wirehair_benchmark.hpp"
#include "xorec_benchmark.hpp"
#include "xorec_gpu_ptr_benchmark.hpp"
#include "xorec_gpu_cmp_benchmark.hpp"



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

void BM_XOREC_GPU_PTR(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XorecBenchmarkGPUPtr>(state, config);
}

void BM_XOREC_GPU_CMP(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XorecBenchmarkGPUCmp>(state, config);
}
