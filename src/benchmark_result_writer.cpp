#include "benchmark_result_writer.h"
#include "utils.h"
#include <benchmark/benchmark.h>
#include <iostream>


BenchmarkCSVReporter::BenchmarkCSVReporter(const std::string& output_file, bool overwrite_file) {
  if (overwrite_file) {
    file.open(output_file, std::ios::out);
  } else {
    file.open(output_file, std::ios::app);
  }

  if (!file.is_open()) {
    throw std::runtime_error("Error opening file: " + output_file);
  }

  if (overwrite_file) {
    file << "plot_id,name,err_msg,num_iterations,time_ns,tot_data_size_B,block_size_B,num_lost_blocks,redundancy_ratio\n";
  }
}

void BenchmarkCSVReporter::ReportRuns(const std::vector<Run>& runs) {
  for (const auto& run : runs) {
    file << static_cast<uint32_t>(run.counters.find("plot_id")->second.value) << ","
         << run.benchmark_name() << ","
         << run.skip_message << ","
         << run.iterations << ","
         << run.GetAdjustedCPUTime() << ","
         << static_cast<uint64_t>(run.counters.find("tot_data_size_B")->second.value) << ","
         << static_cast<uint64_t>(run.counters.find("block_size_B")->second.value) << ","
         << static_cast<uint32_t>(run.counters.find("num_lost_blocks")->second.value) << ","
         << run.counters.find("redundancy_ratio")->second.value << '\n';
  }
}

bool BenchmarkCSVReporter::ReportContext(const Context& context) {
  return true;
}