#ifndef BENCHMARK_RESULT_WRITER_H
#define BENCHMARK_RESULT_WRITER_H

#include <benchmark/benchmark.h>
#include <fstream>
class BenchmarkCSVReporter : public benchmark::BenchmarkReporter {
public:
  explicit BenchmarkCSVReporter(const std::string& output_file, bool overwrite_file);
  ~BenchmarkCSVReporter() {
    if (file.is_open()) {
      file.close();
    }
  }

  void ReportRuns(const std::vector<Run>& runs) override;
  bool ReportContext(const Context& context) override;
private:
  std::ofstream file;
};

#endif // BENCHMARK_RESULT_WRITER_H