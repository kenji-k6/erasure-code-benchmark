/**
 * @file console_reporter.cpp
 * @brief Reporter implementations for displaying the progress bar
 */

#include "console_reporter.hpp"
#include <stdexcept>

ConsoleReporter::ConsoleReporter(int num_runs, std::chrono::system_clock::time_point start_time) : m_bar(num_runs, start_time, std::cout) { }
void ConsoleReporter::update_bar() { m_bar.update(); }
ConsoleReporter::~ConsoleReporter() {}
void ConsoleReporter::ReportRuns([[maybe_unused]] const std::vector<Run>& runs) { return; }
bool ConsoleReporter::ReportContext([[maybe_unused]] const Context& _) {
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