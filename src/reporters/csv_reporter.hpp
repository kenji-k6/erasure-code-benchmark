/**
 * @file csv_reporter.hpp
 * @brief CSV reporter for Google Benchmark
 */

#ifndef CSV_REPORTER_HPP
#define CSV_REPORTER_HPP

#include <benchmark/benchmark.h>
#include <fstream>
#include <string>
#include <string_view>

/**
 * @class CSVReporter
 * @brief Custom benchmark reporter that outputs results in CSV format
 * 
 * This class extends `benchmark::BenchmarkReporter` to write benchmark results
 * to a CSV file. It supports appending to an existing file or overwriting it.
 */
class CSVReporter : public benchmark::BenchmarkReporter {
public:
  /**
   * @brief Construct a new CSVReporter object.
   * @param output_file The path to the output CSV file.
   * @param overwrite_file If true, overwrites the file; otherwise, appends to it.
   * @throws std::runtime_error if the file cannot be opened.
   */
  explicit CSVReporter(const std::string& output_file, bool overwrite_file);

  /**
   * @brief Destructer closes the file stream if open.
   */
  ~CSVReporter() override;

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
  [[nodiscard]] bool ReportContext(const Context& context) override;

protected:
  std::ofstream m_file;         ///< Output file stream for writing benchmark results
  bool m_header_written;        ///< Flag to indicate whether the CSV header has been written.
  bool m_overwrite_file;        ///< Flag to indicate whether to overwrite the file.

  /**
   * @brief Writes the CSV header to the output file.
   */
  virtual void write_header();
};

#endif// CSV_REPORTER_HPP