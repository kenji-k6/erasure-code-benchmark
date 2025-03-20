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



void BM_XOR_BLOCKS_SCALAR(benchmark::State& state, const BenchmarkConfig& config) {
  std::unique_ptr<uint8_t[]> src(new uint8_t[config.block_size]);
  std::unique_ptr<uint8_t[]> dest(new uint8_t[config.block_size]);
  size_t block_size = config.block_size;

  for (auto _ : state) {
    xorec_xor_blocks_scalar(dest.get(), src.get(), block_size);
  }

  state.counters["block_size_B"] = config.block_size;
}

void BM_XOR_BLOCKS_AVX(benchmark::State& state, const BenchmarkConfig& config) {
  std::unique_ptr<uint8_t[]> src(new uint8_t[config.block_size]);
  std::unique_ptr<uint8_t[]> dest(new uint8_t[config.block_size]);
  size_t block_size = config.block_size;

  for (auto _ : state) {
    xorec_xor_blocks_avx(dest.get(), src.get(), block_size);
  }

  state.counters["block_size_B"] = config.block_size;
}

void BM_XOR_BLOCKS_AVX2(benchmark::State& state, const BenchmarkConfig& config) {
  std::unique_ptr<uint8_t[]> src(new uint8_t[config.block_size]);
  std::unique_ptr<uint8_t[]> dest(new uint8_t[config.block_size]);
  size_t block_size = config.block_size;

  for (auto _ : state) {
    xorec_xor_blocks_avx2(dest.get(), src.get(), block_size);
  }

  state.counters["block_size_B"] = config.block_size;
}

void BM_XOR_BLOCKS_AVX512(benchmark::State& state, const BenchmarkConfig& config) {
  std::unique_ptr<uint8_t[]> src(new uint8_t[config.block_size]);
  std::unique_ptr<uint8_t[]> dest(new uint8_t[config.block_size]);
  size_t block_size = config.block_size;

  for (auto _ : state) {
    xorec_xor_blocks_avx512(dest.get(), src.get(), block_size);
  }

  state.counters["block_size_B"] = config.block_size;
}
