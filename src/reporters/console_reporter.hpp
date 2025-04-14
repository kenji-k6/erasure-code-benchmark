#ifndef CONSOLE_REPORTER_HPP
#define CONSOLE_REPORTER_HPP

#include <benchmark/benchmark.h>
#include "progressbar.hpp"

class ConsoleReporter : public benchmark::BenchmarkReporter {
public:
  explicit ConsoleReporter(int num_runs, std::chrono::system_clock::time_point start_time);

  ~ConsoleReporter() override;

  void ReportRuns(const std::vector<Run>& runs) override;

  bool ReportContext(const Context& context) override;

  void update_bar();

protected:
  ProgressBar m_bar;
};
  
#endif // CONSOLE_REPORTER_HPP