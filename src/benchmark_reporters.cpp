#include "benchmark_reporters.h"
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
          << "num_data_blocks,"
          << "num_parity_blocks,"

          << "time_ns,"

          << "encode_time_ns,"
          << "encode_time_ns_stddev,"
          << "encode_throughput_Gbps,"
          << "encode_throughput_Gbps_stddev,"

          << "decode_time_ns,"
          << "decode_time_ns_stddev,"
          << "decode_throughput_Gbps,"
          << "decode_throughput_Gbps_stddev\n"
          << std::flush;
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
          << static_cast<uint32_t>(run.counters.find("num_data_blocks")->second.value) << ","
          << static_cast<uint32_t>(run.counters.find("num_parity_blocks")->second.value) << ","

          << run.GetAdjustedRealTime() << ","

          
          << run.counters.find("encode_time_ns")->second.value << ","
          << run.counters.find("encode_time_ns_stddev")->second.value << ","
          << run.counters.find("encode_throughput_Gbps")->second.value << ","
          << run.counters.find("encode_throughput_Gbps_stddev")->second.value << ","


          << run.counters.find("decode_time_ns")->second.value << ","
          << run.counters.find("decode_time_ns_stddev")->second.value << ","
          << run.counters.find("decode_throughput_Gbps")->second.value << ","
          << run.counters.find("decode_throughput_Gbps_stddev")->second.value << "\n"
          << std::flush;
  }
}

bool BenchmarkCSVReporter::ReportContext(const Context& _) { return true; }




BenchmarkProgressReporter::BenchmarkProgressReporter(int num_runs) : bar_(num_runs, std::cout) { bar_.update(); }
void BenchmarkProgressReporter::update_bar() { bar_.update(); }
BenchmarkProgressReporter::~BenchmarkProgressReporter() {}
void BenchmarkProgressReporter::ReportRuns(const std::vector<Run>& runs) { return; }
bool BenchmarkProgressReporter::ReportContext(const Context& _) { return true; }