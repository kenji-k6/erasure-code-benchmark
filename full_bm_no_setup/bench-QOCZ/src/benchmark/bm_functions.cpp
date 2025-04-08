/**
 * @file bm_functions.cpp
 * @brief Implements benchmark functions for different EC algorithms.
 */

#include "bm_functions.hpp"
#include "bm_generic_runner.hpp"
#include "cm256_bm.hpp"
#include "isal_bm.hpp"
#include "leopard_bm.hpp"
#include "xorec_avx_bm.hpp"
#include "xorec_avx2_bm.hpp"
#include "xorec_avx512_bm.hpp"



void BM_CM256(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<CM256Benchmark>(state, config);
}

void BM_ISAL(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<ISALBenchmark>(state, config);
}

void BM_Leopard(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<LeopardBenchmark>(state, config);
}

void BM_XOREC_AVX(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XorecBenchmarkAVX>(state, config);
}

void BM_XOREC_AVX2(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XorecBenchmarkAVX2>(state, config);
}

void BM_XOREC_AVX512(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<XorecBenchmarkAVX512>(state, config);
}