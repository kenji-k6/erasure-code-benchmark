/**
 * @file bm_reporters.cpp
 * @brief Reporter implementations for writing to the CSV output file and displaying the progress bar
 */


#include "bm_reporters.hpp"
#include <stdexcept>


BenchmarkCSVReporter::BenchmarkCSVReporter(const std::string& output_file, bool overwrite_file, size_t num_repetitions)
    : m_header_written(false), 
      m_overwrite_file(overwrite_file),
      m_num_repetitions(num_repetitions) 
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

  if (runs.size() == m_num_repetitions) return; // skip if the runs aren't aggregates

  auto mean_run = runs[0];
  auto stddev_run = runs[2];
  auto pos = mean_run.benchmark_name().find("_mean");

  std::string name = "\"" + mean_run.benchmark_name().substr(0, pos) + "\"";
  bool data_corrupted = static_cast<bool>(mean_run.counters.find("data_corrupted")->second.value);
  std::string err_msg = data_corrupted ? "\"Corruption Detected!\"" : "\"\"";
  uint64_t num_iterations = mean_run.iterations;
  uint64_t message_size_B = static_cast<uint64_t>(mean_run.counters.find("message_size_B")->second.value);
  uint64_t block_size_B = static_cast<uint64_t>(mean_run.counters.find("block_size_B")->second.value);

  uint32_t fec_0 = static_cast<uint32_t>(mean_run.counters.find("fec_params_0")->second.value);
  uint32_t fec_1 = static_cast<uint32_t>(mean_run.counters.find("fec_params_1")->second.value);
  std::string fec = "\"FEC(" +std::to_string(fec_0) + "," + std::to_string(fec_1) + ")\"";
  uint32_t num_lost_rdma_packets = static_cast<uint32_t>(mean_run.counters.find("num_lost_rdma_packets")->second.value);
  uint32_t is_gpu_bm = static_cast<uint32_t>(mean_run.counters.find("is_gpu_bm")->second.value);
  uint32_t num_gpu_blocks = static_cast<uint32_t>(mean_run.counters.find("num_gpu_blocks")->second.value);
  uint32_t threads_per_gpu_block = static_cast<uint32_t>(mean_run.counters.find("threads_per_gpu_block")->second.value);
  double time_ns = mean_run.GetAdjustedCPUTime();
  double time_ns_stddev = stddev_run.GetAdjustedCPUTime();

  m_file << name                  << ","
         << err_msg               << ","
         << num_iterations        << ","
         << message_size_B        << ","
         << block_size_B          << ","
         << fec                   << ","
         << num_lost_rdma_packets << ","
         << is_gpu_bm             << ","
         << num_gpu_blocks        << ","
         << threads_per_gpu_block << ","
         << time_ns               << ","
         << time_ns_stddev        << std::endl;
}

bool BenchmarkCSVReporter::ReportContext([[maybe_unused]]const Context& _) { return true; }

void BenchmarkCSVReporter::write_header() {
  m_file  << "name,"
          << "err_msg,"
          << "num_iterations,"
          << "message_size_B,"
          << "block_size_B,"
          << "FEC,"
          << "num_lost_rdma_packets,"
          << "is_gpu_bm,"
          << "num_gpu_blocks,"
          << "threads_per_gpu_block,"
          << "time_ns,"
          << "time_ns_stddev"
          << std::endl;
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