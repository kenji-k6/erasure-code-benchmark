#include "benchmark_result_writer.h"
#include <stdexcept>



BenchmarkCSVReporter::BenchmarkCSVReporter(const std::string& output_file, bool overwrite_file) {
  std::ios_base::openmode mode = overwrite_file ? std::ios::out : std::ios::app;
  file.open(output_file, mode);

  if (!file.is_open()) {
    throw std::runtime_error("Error opening file: " + output_file);
  }

  // Write CSV header only if overwriting the file
  if (overwrite_file) {
    file  << "plot_id,"
          << "name,"
          << "err_msg,"
          << "num_iterations,"
          << "tot_data_size_B,"
          << "block_size_B,"
          << "num_lost_blocks,"
          << "redundancy_ratio,"

          << "time_ns,"
          << "tot_time_stddev_ns,"
          << "tot_time_min_ns,"
          << "tot_time_max_ns,"

          << "encode_time_ns,"
          << "encode_time_stddev_ns,"
          << "encode_time_min_ns,"
          << "encode_time_max_ns,"

          << "decode_time_ns,"
          << "encode_time_stddev_ns,"
          << "decode_time_min_ns,"
          << "decode_time_max_ns\n";
          
  }
}

BenchmarkCSVReporter::~BenchmarkCSVReporter() {
  if (file.is_open()) file.close();
}

void BenchmarkCSVReporter::ReportRuns(const std::vector<Run>& runs) {
  for (const auto& run : runs) {
    file  << static_cast<uint32_t>(run.counters.find("plot_id")->second.value) << ","
          << run.benchmark_name() << ","
          << run.skip_message << ","
          << run.iterations << ","
          << static_cast<uint64_t>(run.counters.find("tot_data_size_B")->second.value) << "," 
          << static_cast<uint64_t>(run.counters.find("block_size_B")->second.value) << ","
          << static_cast<uint32_t>(run.counters.find("num_lost_blocks")->second.value) << ","
          << run.counters.find("redundancy_ratio")->second.value << ","

          << run.GetAdjustedRealTime() << ","
          << run.counters.find("tot_time_stddev_ns")->second.value << ","
          << run.counters.find("tot_time_min_ns")->second.value << ","
          << run.counters.find("tot_time_max_ns")->second.value << ","
          
          << run.counters.find("encode_time_ns")->second.value << ","
          << run.counters.find("encode_time_stddev_ns")->second.value << ","
          << run.counters.find("encode_time_min_ns")->second.value << ","
          << run.counters.find("encode_time_max_ns")->second.value << ","

          << run.counters.find("decode_time_ns")->second.value << ","
          << run.counters.find("decode_time_stddev_ns")->second.value << ","
          << run.counters.find("decode_time_min_ns")->second.value << ","
          << run.counters.find("decode_time_max_ns")->second.value << "\n";
  }
}

bool BenchmarkCSVReporter::ReportContext(const Context& context) { return true; }