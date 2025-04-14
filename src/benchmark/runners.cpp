/**
 * @file runners.cpp
 * @brief Implements benchmark functions for different EC algorithms.
 */

#include "runners.hpp"
#include "abstract_runner.hpp"
#include "cm256_bm.hpp"
#include "isal_bm.hpp"
#include "leopard_bm.hpp"
#include "xorec_bm.hpp"
#include "xorec_gpu_ptr_bm.hpp"
#include "xorec_unified_ptr_bm.hpp"
#include "xorec_gpu_cmp_bm.hpp"
#include "xorec.hpp"



void BM_CM256(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<CM256Benchmark>(state, config);
}

void BM_ISAL(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<ISALBenchmark>(state, config);
}

void BM_Leopard(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<LeopardBenchmark>(state, config);
}

void BM_XOREC(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XorecBenchmark>(state, config);
}

void BM_XOREC_UNIFIED_PTR(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XorecBenchmarkUnifiedPtr>(state, config);
}

void BM_XOREC_GPU_PTR(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XorecBenchmarkGpuPtr>(state, config);
}

void BM_XOREC_GPU_CMP(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XorecBenchmarkGpuCmp>(state, config);
}
