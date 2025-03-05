#include "benchmark_utils.h"

constexpr const char* OUTPUT_FILE_DIR = "../results/raw/";

int main (int argc, char** argv) {
  std::vector<BenchmarkConfig> configs;
  std::vector<uint32_t> lost_block_idxs;

  get_configs(argc, argv, configs, lost_block_idxs);

  if (configs.empty()) exit(1);

  int tot_num_iterations = configs[0].num_iterations * configs.size() * available_benchmarks.size();
  BenchmarkProgressReporter console_reporter()

}