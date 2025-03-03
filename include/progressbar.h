#ifndef PROGRESSBAR_H
#define PROGRESSBAR_H

#include <iostream>
#include <ostream>
#include <string>
#include <stdexcept>
#include <chrono>
#include <iomanip>


/**
 * @class ProgressBar
 * @brief A simple command-line progress bar utility.
 *
 * This class provides a visual progress bar in the terminal to track iterative processes.
 * It prints progress updates dynamically using ANSI escape codes.
 */
class ProgressBar {
public:
  /**
   * @brief Constructs a ProgressBar with a given number of iterations.
   * @param total_steps The total number of iterations (must be > 0).
   * @param out The output stream to print the progress bar (default: std::cerr).
   */
  explicit ProgressBar(int total_steps, std::ostream& out=std::cerr)
    : total_steps_(total_steps),
      output_(out),
      start_time_(std::chrono::system_clock::now()) {
    
    if (total_steps_ <= 0) throw std::invalid_argument("Number of iterations must be > 0");
    }
  

  // Deleting copy and move operations
  ProgressBar(const ProgressBar&) = delete;
  ProgressBar& operator=(const ProgressBar&) = delete;
  ProgressBar(ProgressBar&&) = delete;
  ProgressBar& operator=(ProgressBar&&) = delete;


  /**
   * @brief Resets the progress bar state.
   */
  void reset() {
    current_step_ = 0;
    last_percentage_ = 0;
    first_update_called_ = false;
  }


  /**
   * @brief Updates the progress bar with the current step.
   */
  void update() {
    if (!first_update_called_) initialize_bar();

    int percentage = (current_step_ * 100) / (total_steps_ - 1);
    if (percentage < last_percentage_) return;

    redraw_bar(percentage);
    print_elapsed_time();

    last_percentage_ = percentage;
    ++current_step_;
    output_ << std::flush;
  }


private:
  int current_step_ = 0;                              ///< Current progress step.
  int total_steps_;                                   ///< Total steps in the process.
  int last_percentage_ = 0;                           ///< Last printed percentage to avoid redundant updates.
  bool first_update_called_ = false;                  ///< Flag to check if the first update is called.

  const std::string done_char_ = "█";                 ///< Character for completed part of the progress bar.
  const std::string todo_char_ = " ";                 ///< Character for remaining part of the progress bar.
  const std::string open_bracket_ = "[";              ///< Left boundary of the progress bar.
  const std::string close_bracket_ = "]";             ///< Right boundary of the progress bar.

  std::ostream& output_;                              ///< Output stream for the progress bar.
  std::chrono::system_clock::time_point start_time_;  ///< Start time for elapsed time tracking.


  /**
   * @brief Initializes the progress bar on the first udate call.
   */
  void initialize_bar() {
    output_ << '\n' << open_bracket_;
    for (int i = 0; i < 50; ++i) output_ << todo_char_;
    output_ << close_bracket_ << "\033[1;39m" << " 0%" << "\033[0m" << '\n';
    first_update_called_ = true;
    start_time_ = std::chrono::system_clock::now();
  }


  /**
   * @brief Redraws the progress bar with a given percentage.
   * @param percentage The progress percentage to display
   */
  void redraw_bar(int percentage) {
    output_ << "\033[A\033[K\033[A\033[K\033[A\033[K\n"; // Clear previous output
    output_ << open_bracket_;

    // Compute filled portion of the progress bar
    int completed = percentage / 2;

    output_ << "\033[32m"; // Green color for completed part
    for (int i = 0; i < completed; ++i) output_ << done_char_;
    output_ << "\033[0m"; // Reset color

    for (int i = completed; i < 50; ++i) output_ << todo_char_;
    
    output_ << close_bracket_           // Close progress bar
            << "\033[1;39m"             // Bold text for percentage
            << ' ' << percentage << '%' // Print percentage
            << "\033[0m"                // Reset text style
            << '\n';                    // Move to next line
  
  }


  /**
   * @brief Prints the elapsed time since the progress bar started.
   */
  void print_elapsed_time() {
    auto now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_);

    int hours = static_cast<int>(duration.count() / 3600);
    int minutes = static_cast<int>((duration.count() % 3600) / 60);
    int seconds = static_cast<int>(duration.count() % 60);

    output_ << "\033[1;31m" << "Elapsed time: " << "\033[33m"
            << std::setw(2) << std::setfill('0') << hours << ':'
            << std::setw(2) << std::setfill('0') << minutes << ':'
            << std::setw(2) << std::setfill('0') << seconds
            << "\033[0m" << '\n';
  }
};

#endif // PROGRESSBAR_H