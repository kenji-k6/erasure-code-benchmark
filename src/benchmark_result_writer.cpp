#include "benchmark_result_writer.h"
#include <benchmark/benchmark.h>
#include <iostream>


explicit BenchmarkCSVReporter::BenchmarkCSVReporter(const std::string& output_file) : file(output_file) {
  if (!file.is_open()) {
    throw std::runtime_error("Error opening file: " + output_file);
  }

  // Define the custom header columns
  file << "name,iterations,real_time,cpu_time,error_message,tot_bytes,block_bytes,num_lost_blocks,redundancy_ratio\n";
}