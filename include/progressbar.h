#ifndef PROGRESSBAR_H
#define PROGRESSBAR_H
#include <iostream>
#include <ostream>
// #include <iomanip>
#include <string>
#include <stdexcept>
#include <chrono>

class ProgressBar {
public:
  ~ProgressBar() = default;

  ProgressBar (ProgressBar const&) = delete;
  ProgressBar& operator=(ProgressBar const&) = delete;
  ProgressBar (ProgressBar&&) = delete;
  ProgressBar& operator=(ProgressBar&&) = delete;

  inline ProgressBar(int n_iter, std:: ostream& out=std::cerr);

  inline void reset();
  inline void update();
  inline void update_elapsed_time();

private:
  int progress;
  int n_cycles;
  int last_perc;
  bool update_is_called;

  std::string done_char;
  std::string todo_char;
  std::string opening_bracket_char;
  std::string closing_bracket_char;

  std::chrono::system_clock::time_point start_time;

  std::ostream& output;
};

inline ProgressBar::ProgressBar(int n_iter, std::ostream& out) :
  progress(0),
  n_cycles(n_iter),
  last_perc(0),
  update_is_called(false),
  done_char("█"),
  todo_char(" "),
  opening_bracket_char("["),
  closing_bracket_char("]"),
  output(out) {}

inline void ProgressBar::reset() {
  progress = 0;
  update_is_called = false;
  last_perc = 0;
  return;
}

inline void ProgressBar::update() {
  if (n_cycles <= 0) throw std::runtime_error("Number of iterations is <= 0");

  if (!update_is_called) {
    // First time update is called
    output << '\n' << opening_bracket_char;
    for (int _ = 0; _ < 50; ++_) output << todo_char;
    output << closing_bracket_char << "\033[1;39m" << " 0%" << "\033[0m" << '\n';
    update_is_called = true;
    start_time = std::chrono::system_clock::now();
  }

  // compute percentage
  int perc = progress*100.0/(n_cycles-1);
  if (perc < last_perc) return;

  // erase elapsed time, progress bar and percentage
  output << "\033[A\033[K\033[A\033[K\033[A\033[K\n";

  // update progress bar and percentage
  int bar_perc = (perc%2 == 0) ? perc : perc-1;
  
  output << opening_bracket_char;
  // change color for completed part of the bar
  output << "\033[32m";

  for (int j = 0; j < (bar_perc)/2; ++j) {
    output << done_char;
  }
  //Reset color
  output << "\033[0m";

  for (int j = 0; j < 50-(bar_perc)/2; ++j) {
    output << todo_char;
  }

  output  << closing_bracket_char // close progress bar
          << "\033[1;39m" // make text bold for percentage
          << ' ' << perc << '%' // print percentage
          << "\033[0m" // reset text style
          << '\n'; // move to next line
  
  // update elapsed time
  update_elapsed_time();

  last_perc = perc;
  ++progress;
  output << std::flush;
  return;
}

inline void ProgressBar::update_elapsed_time() {
  auto now = std::chrono::system_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);

  auto hours = duration_cast<std::chrono::hours>(duration);
  duration -= hours;
  auto minutes = duration_cast<std::chrono::minutes>(duration);
  duration -= minutes;
  auto seconds = duration_cast<std::chrono::seconds>(duration);
  
  output  << "\033[1;31m" << "Elapsed time: " << "\033[33m"
          << std::setw(2) << std::setfill('0') << hours.count() << ':'
          << std::setw(2) << std::setfill('0') << minutes.count() << ':'
          << std::setw(2) << std::setfill('0') << seconds.count() << "\033[0m" << '\n';
}
#endif // PROGRESSBAR_H