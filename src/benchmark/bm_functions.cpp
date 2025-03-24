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

void BM_Wirehair(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<WirehairBenchmark>(state, config);
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



static void perf_bm_set_counters(benchmark::State& state, const BenchmarkConfig& config) {
  state.counters["block_size_B"] = config.block_size;
  state.counters["xorec_version"] = static_cast<int>(config.xorec_params.version);
}

void BM_XOR_BLOCKS_SCALAR(benchmark::State& state, const BenchmarkConfig& config) {
  uint8_t *src = reinterpret_cast<uint8_t*>(malloc(config.block_size));
  uint8_t *dest = reinterpret_cast<uint8_t*>(malloc(config.block_size));
  size_t block_size = config.block_size;

  for (auto _ : state) {
    xorec_xor_blocks_scalar(dest, src, block_size);
  }

  perf_bm_set_counters(state, config);
}

void BM_XOR_BLOCKS_AVX(benchmark::State& state, const BenchmarkConfig& config) {
  uint8_t *src = reinterpret_cast<uint8_t*>(_mm_malloc(config.block_size, 64));
  uint8_t *dest = reinterpret_cast<uint8_t*>(_mm_malloc(config.block_size, 64));
  size_t block_size = config.block_size;

  for (auto _ : state) {
    xorec_xor_blocks_avx(dest, src, block_size);
  }

  perf_bm_set_counters(state, config);
  _mm_free(src);
  _mm_free(dest);
}

void BM_XOR_BLOCKS_AVX2(benchmark::State& state, const BenchmarkConfig& config) {
  uint8_t *src = reinterpret_cast<uint8_t*>(_mm_malloc(config.block_size, 64));
  uint8_t *dest = reinterpret_cast<uint8_t*>(_mm_malloc(config.block_size, 64));
  size_t block_size = config.block_size;

  for (auto _ : state) {
    xorec_xor_blocks_avx2(dest, src, block_size);
  }

  perf_bm_set_counters(state, config);
  _mm_free(src);
  _mm_free(dest);
}

void BM_XOR_BLOCKS_AVX512(benchmark::State& state, const BenchmarkConfig& config) {
  uint8_t *src = reinterpret_cast<uint8_t*>(_mm_malloc(config.block_size, 64));
  uint8_t *dest = reinterpret_cast<uint8_t*>(_mm_malloc(config.block_size, 64));
  size_t block_size = config.block_size;

  for (auto _ : state) {
    xorec_xor_blocks_avx512(dest, src, block_size);
  }

  perf_bm_set_counters(state, config);
  _mm_free(src);
  _mm_free(dest);
}
