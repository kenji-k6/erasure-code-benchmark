#include "bm_utils.hpp"

int main (int argc, char** argv) {
  std::vector<BenchmarkConfig> ec_configs;
  std::vector<BenchmarkConfig> perf_configs;
  std::vector<std::vector<uint32_t>> lost_block_idxs;
  
  parse_args(argc, argv);
  get_configs(ec_configs, lost_block_idxs, perf_configs);
  run_benchmarks(ec_configs, perf_configs);
  
}