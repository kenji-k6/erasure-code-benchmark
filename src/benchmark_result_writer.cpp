#include "benchmark_result_writer.h"
#include "utils.h"
#include <benchmark/benchmark.h>
#include <iostream>


BenchmarkCSVReporter::BenchmarkCSVReporter(const std::string& output_file) : file(output_file) {
  if (!file.is_open()) {
    std::cout << "test\n";
    throw std::runtime_error("Error opening file: " + output_file);
  }

  // Define the custom header columns
  file << "name,iterations,real_time,cpu_time,err_msg,tot_bytes,block_bytes,num_lost_blocks,redundancy_ratio\n";
}

void BenchmarkCSVReporter::ReportRuns(const std::vector<Run>& runs) {
  uint32_t tot_bytes = benchmark_config.data_size;
  uint32_t block_bytes = benchmark_config.block_size;
  uint32_t num_lost_blocks = benchmark_config.num_lost_blocks;
  double redundancy_ratio = benchmark_config.redundancy_ratio;

  for (const auto& run : runs) {
    file << run.benchmark_name() << ","
         << run.iterations << ","
         << run.GetAdjustedRealTime() << ","
         << run.GetAdjustedCPUTime() << ","
         << run.skip_message << ","
         << tot_bytes << ","
         << block_bytes << ","
         << num_lost_blocks << ","
         << redundancy_ratio << "\n";
  }
}

bool BenchmarkCSVReporter::ReportContext(const Context& context) {
  return true;
}