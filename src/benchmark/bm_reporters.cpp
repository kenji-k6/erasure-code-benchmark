/**
 * @file bm_reporters.cpp
 * @brief Reporter implementations for writing to the CSV output file and displaying the progress bar
 */


#include "bm_reporters.hpp"
#include <stdexcept>


BenchmarkCSVReporter::BenchmarkCSVReporter(const std::string& output_file, bool overwrite_file)
    : m_header_written(false), 
      m_overwrite_file(overwrite_file) 
{
  std::ios_base::openmode mode = overwrite_file ? std::ios::out : std::ios::app;
  m_file.open(output_file, mode);

  if (!m_file.is_open()) {
    throw std::runtime_error("Error opening file: " + output_file);
  }
}

BenchmarkCSVReporter::~BenchmarkCSVReporter() {
  if (m_file.is_open()) m_file.close();
}

void BenchmarkCSVReporter::ReportRuns(const std::vector<Run>& runs) {
  // Write CSV header only if overwriting the file and it hasn't been written yet
  if (m_overwrite_file && !m_header_written) {
    write_header();
    m_header_written = true;
  }

  for (const auto& run : runs) {
    m_file  << static_cast<uint32_t>(run.counters.find("plot_id")->second.value) << ","
            << "\"" << run.benchmark_name() << "\","
            << run.skip_message << ","
            << run.iterations << ","
            << static_cast<uint64_t>(run.counters.find("message_size_B")->second.value) << "," 
            << static_cast<uint64_t>(run.counters.find("block_size_B")->second.value) << ","
            << static_cast<uint32_t>(run.counters.find("fec_params_0")->second.value) << ","
            << static_cast<uint32_t>(run.counters.find("fec_params_1")->second.value) << ","
            << static_cast<uint32_t>(run.counters.find("num_lost_rdma_packets")->second.value) << ","
            << static_cast<uint32_t>(run.counters.find("is_gpu_bm")->second.value) << ","
            << static_cast<uint32_t>(run.counters.find("num_gpu_blocks")->second.value) << ","
            << static_cast<uint32_t>(run.counters.find("threads_per_gpu_block")->second.value) << ","
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

bool BenchmarkCSVReporter::ReportContext([[maybe_unused]]const Context& _) { return true; }

void BenchmarkCSVReporter::write_header() {
  m_file  << "name,"
          << "err_msg,"
          << "num_iterations,"
          << "message_size_B,"
          << "block_size_B,"
          << "fec_params_0,"
          << "fec_params_1,"
          << "num_lost_rdma_packets,"
          << "is_gpu_bm,"
          << "num_gpu_blocks,"
          << "threads_per_gpu_block,"
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




BenchmarkProgressReporter::BenchmarkProgressReporter(int num_runs, std::chrono::system_clock::time_point start_time) : m_bar(num_runs, start_time, std::cout) { }
void BenchmarkProgressReporter::update_bar() { m_bar.update(); }
BenchmarkProgressReporter::~BenchmarkProgressReporter() {}
void BenchmarkProgressReporter::ReportRuns([[maybe_unused]] const std::vector<Run>& runs) { return; }
bool BenchmarkProgressReporter::ReportContext([[maybe_unused]] const Context& _) {
  #if defined(__GNUC__)
      std::cout << "Compiler: GCC\n";
      std::cout << "Version: " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__ << "\n\n";
  #elif defined(__clang__)
      std::cout << "Compiler: Clang\n";
      std::cout << "Version: " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__ << "\n";
  #elif defined(__INTEL_COMPILER)
      std::cout << "Compiler: Intel Compiler\n";
      std::cout << "Version: " << __INTEL_COMPILER << "\n";
  #else
      std::cout << "Unknown Compiler\n";
  #endif
  m_bar.update();
  return true; 
}