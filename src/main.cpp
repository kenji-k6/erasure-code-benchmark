#include "benchmark_utils.hpp"

int main (int argc, char** argv) {
  std::vector<BenchmarkConfig> configs;
  std::vector<std::vector<uint32_t>> lost_block_idxs;
  
  parse_args(argc, argv);
  get_configs(configs, lost_block_idxs);
  run_benchmarks(configs);
}