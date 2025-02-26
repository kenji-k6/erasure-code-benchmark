#ifndef BENCHMARK_RESULT_WRITER_H
#define BENCHMARK_RESULT_WRITER_H

#include <benchmark/benchmark.h>
#include <fstream>
class BenchmarkCSVReporter : public benchmark::BenchmarkReporter {
public:
  explicit BenchmarkCSVReporter(const std::string& output_file);

  void ReportRuns(const std::vector<Run>& runs) override;

private:
  std::ofstream file;
};

#endif // BENCHMARK_RESULT_WRITER_H