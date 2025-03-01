#ifndef BENCHMARK_REPORTERS_H
#define BENCHMARK_REPORTERS_H

#include <benchmark/benchmark.h>
#include <fstream>
#include <string>

/**
 * @class BenchmarkCSVReporter
 * @brief Custom benchmark reporter that outputs results in CSV format.
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

private:
  std::ofstream file; ///< Output file stream for writing benchmark results.
};

#endif // BENCHMARK_REPORTERS_H