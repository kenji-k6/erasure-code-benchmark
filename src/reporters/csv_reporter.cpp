/**
 * @file csv_reporter.cpp
 * @brief Reporter implementations for writing to the CSV output file
 */

#include "csv_reporter.hpp"
#include <stdexcept>
#include <string_view>
#include <ranges>

CSVReporter::CSVReporter(const std::string& output_file, bool overwrite_file)
  : m_header_written(!overwrite_file), 
    m_overwrite_file(overwrite_file)
{
  std::ios_base::openmode mode = overwrite_file ? std::ios::out : std::ios::app;
  m_file.open(output_file, mode);

  if (!m_file.is_open()) throw std::runtime_error("Error opening file: " + output_file);
}

CSVReporter::~CSVReporter() {
  if (m_file.is_open()) m_file.close();
}

void CSVReporter::write_header() {
  constexpr std::string_view header =
  "name,err_msg,iterations,warmup_iterations,"
  "gpu_computation,gpu_blocks,threads_per_block,"
  "message_size_B,block_size_B,EC,lost_blocks,cpu_threads,"
  "encode_time_ns,encode_time_ns_stddev,"
  "encode_throughput_Gbps,encode_throughput_Gbps_stddev,"
  "decode_time_ns,decode_time_ns_stddev,"
  "decode_throughput_Gbps,decode_throughput_Gbps_stddev\n";
  m_file << header << std::flush;
}

void CSVReporter::ReportRuns(const std::vector<Run>& runs) {
  if (runs.size() != 1) throw std::runtime_error("Error during CSV report generation");

  if (!m_header_written) {
    write_header();
    m_header_written = true;
  }

  const auto& run = runs[0];
  const auto& counters = run.counters;

  auto get_counter = [&counters](const std::string& key) -> double { return counters.find(key)->second.value; };

  auto benchmark_name = run.benchmark_name();
  auto cut_off = benchmark_name.find_first_of("/");
  std::string name = "\"" + std::string(benchmark_name.substr(0, cut_off)) + "\"";

  int warmup_iterations = static_cast<int>(get_counter("num_warmup_iterations"));
  int cpu_threads = static_cast<int>(get_counter("num_cpu_threads"));
  int gpu_computation = static_cast<int>(get_counter("gpu_computation"));
  int gpu_blocks = static_cast<int>(get_counter("num_gpu_blocks"));
  int threads_per_block = static_cast<int>(get_counter("threads_per_gpu_block"));
  size_t message_size_B = static_cast<size_t>(get_counter("message_size_B"));
  size_t block_size_B = static_cast<size_t>(get_counter("block_size_B"));
  size_t lost_blocks = static_cast<size_t>(get_counter("num_lost_blocks"));

  int ec_0 = static_cast<int>(get_counter("ec_params_0"));
  int ec_1 = static_cast<int>(get_counter("ec_params_1"));
  std::string ec_params = "\"(" + std::to_string(ec_0) + "/" + std::to_string(ec_1) + ")\"";

  double enc_t_ns = get_counter("encode_time_ns");
  double enc_t_ns_stddev = get_counter("encode_time_ns_stddev");
  double enc_tp_Gbps = get_counter("encode_throughput_Gbps");
  double enc_tp_Gbps_stddev = get_counter("encode_throughput_Gbps_stddev");

  double dec_t_ns = get_counter("decode_time_ns");
  double dec_t_ns_stddev = get_counter("decode_time_ns_stddev");
  double dec_tp_Gbps = get_counter("decode_throughput_Gbps");
  double dec_tp_Gbps_stddev = get_counter("decode_throughput_Gbps_stddev");

  m_file << name << ','
         << run.skip_message << ","
         << run.iterations << ","
         << warmup_iterations << ","
         << gpu_computation << ","
         << gpu_blocks << ","
         << threads_per_block << ","
         << message_size_B << ","
         << block_size_B << ","
         << ec_params << ","
         << lost_blocks << ","
         << cpu_threads << ","
         << enc_t_ns << ","
         << enc_t_ns_stddev << ","
         << enc_tp_Gbps << ","
         << enc_tp_Gbps_stddev << ","
         << dec_t_ns << ","
         << dec_t_ns_stddev << ","
         << dec_tp_Gbps << ","
         << dec_tp_Gbps_stddev
         << "\n" 
         << std::flush;
}

bool CSVReporter::ReportContext([[maybe_unused]]const Context& _) { return true; }