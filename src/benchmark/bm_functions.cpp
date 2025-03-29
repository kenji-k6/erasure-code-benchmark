/**
 * @file bm_functions.cpp
 * @brief Implements benchmark functions for different EC algorithms.
 */

#include "bm_functions.hpp"
#include "bm_generic_runner.hpp"
#include "cm256_bm.hpp"
#include "isal_bm.hpp"
#include "leopard_bm.hpp"
#include "wirehair_bm.hpp"
#include "xorec_bm.hpp"
#include "xorec_gpu_cmp_bm.hpp"



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

void BM_XOREC_GPU(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XorecBenchmarkGpuCmp>(state, config);
}

void BM_XOREC_GPU_PARITY_CPU(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XorecBenchmarkGpuCmpCpuParity>(state, config);
}