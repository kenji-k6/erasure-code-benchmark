#include "benchmark_utils.h"
#include "xorec_gpu.h"

int main (int argc, char** argv) {
  std::vector<BenchmarkConfig> configs;
  std::vector<std::vector<uint32_t>> lost_block_idxs;

  get_configs(argc, argv, configs, lost_block_idxs);
  run_benchmarks(configs);
}