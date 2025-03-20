#ifndef BM_REPORTERS_HPP
#define BM_REPORTERS_HPP

#include <benchmark/benchmark.h>
#include "progressbar.hpp"
#include <fstream>

/**
 * @class ECBenchmarkCSVReporter
 * @brief Custom benchmark reporter that outputs results in CSV format. (used for EC benchmarks)
 * 
 * This class extends `benchmark::BenchmarkReporter` to write benchmark results
 * to a CSV file. It supports appending to an existing file or overwriting it.
 */
class BenchmarkCSVReporter : public benchmark::BenchmarkReporter {
public:
  /**
   * @brief Construct a new BenchmarkCSVReporter object.
   * @param output_file The path to the output CSV file.
   * @param overwrite_file If true, overwrites the file; otherwise, appends to it.
   * @throwss std::runtime_error if the file cannot be opened.
   */
  explicit BenchmarkCSVReporter(const std::string& output_file, bool overwrite_file);

  /**
   * @brief Destructer closes the file stream if open.
   */
  ~BenchmarkCSVReporter() override;

  /**
   * @brief Reports a set of benchmark runs to the CSV file.
   * @param runs A vector of benchmark run results.
   */
  void ReportRuns(const std::vector<Run>& runs) override;

  /**
   * @brief Reports the benchmark context
   * @param context The benchmark context.
   * @return Always returns true. Unused for our purposes.
   */
  bool ReportContext(const Context& context) override;

protected:
  std::ofstream m_file;         ///< Output file stream for writing benchmark results
  bool m_header_written;        ///< Flag to indicate whether the CSV header has been written.
  bool m_overwrite_file;        ///< Flag to indicate whether to overwrite the file.
  virtual void write_header();
};



class BenchmarkProgressReporter : public benchmark::BenchmarkReporter {
public:
  explicit BenchmarkProgressReporter(int num_runs, std::chrono::system_clock::time_point start_time);

  ~BenchmarkProgressReporter() override;

  void ReportRuns(const std::vector<Run>& runs) override;

  bool ReportContext(const Context& context) override;

  void update_bar();

protected:
  ProgressBar m_bar;
};


/**
 * @class PerfBenchmarkCSVReporter
 * @brief Custom benchmark reporter that outputs results in CSV format. (used for performance benchmarks)
 * 
 * This class extends `benchmark::BenchmarkReporter` to write benchmark results
 * to a CSV file. It supports appending to an existing file or overwriting it.
 */
class PerfBenchmarkCSVReporter : public BenchmarkCSVReporter {
public:
  explicit PerfBenchmarkCSVReporter(const std::string& output_file, bool overwrite_file) : BenchmarkCSVReporter(output_file, overwrite_file) { };
  void ReportRuns(const std::vector<Run>& runs) override;

protected:
  void write_header() override;
};


class PerfBenchmarkProgressReporter : public BenchmarkProgressReporter {
public:
  explicit PerfBenchmarkProgressReporter(int num_runs, std::chrono::system_clock::time_point start_time) : BenchmarkProgressReporter(num_runs, start_time) { };
  void ReportRuns(const std::vector<Run>& runs) override;
  bool ReportContext(const Context& context) override;
};


#endif // BM_REPORTERS_HPP