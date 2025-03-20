#ifndef PROGRESSBAR_HPP
#define PROGRESSBAR_HPP

#include <iostream>
#include <ostream>
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
  explicit ProgressBar(int total_steps, std::chrono::system_clock::time_point start_time=std::chrono::system_clock::now(), std::ostream& out=std::cerr)
    : m_total_steps(total_steps),
      m_start_time(start_time),
      m_output(out) {
    
    if (m_total_steps <= 0) throw std::invalid_argument("Number of iterations must be > 0");
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
    m_current_step = 0;
    m_last_percentage = 0;
    m_first_update_called = false;
  }


  /**
   * @brief Updates the progress bar with the current step.
   */
  void update() {
    if (!m_first_update_called) {
      initialize_bar();
      m_first_update_called = true;
    } else {
      ++m_current_step;
    }

    int percentage = (m_current_step * 100) / (m_total_steps - 1);
    if (percentage < m_last_percentage) return;
    if (percentage > 100) percentage = 100;

    redraw_bar(percentage);
    print_elapsed_time();

    m_last_percentage = percentage;
    m_output << std::flush;
  }


private:
  int m_current_step = 0;                              ///< Current progress step.
  int m_total_steps;                                   ///< Total steps in the process.
  int m_last_percentage = 0;                           ///< Last printed percentage to avoid redundant updates.
  bool m_first_update_called = false;                  ///< Flag to check if the first update is called.

  const std::string m_done_char = "█";                 ///< Character for completed part of the progress bar.
  const std::string m_todo_char = " ";                 ///< Character for remaining part of the progress bar.
  const std::string m_open_bracket = "[";              ///< Left boundary of the progress bar.
  const std::string m_close_bracket = "]";             ///< Right boundary of the progress bar.

  std::chrono::system_clock::time_point m_start_time;  ///< Start time for elapsed time tracking.
  std::ostream& m_output;                              ///< Output stream for the progress bar.


  /**
   * @brief Initializes the progress bar on the first udate call.
   */
  void initialize_bar() {
    m_output << '\n' << m_open_bracket;
    for (int i = 0; i < 50; ++i) m_output << m_todo_char;
    m_output << m_close_bracket << "\033[1;31m" << " 0%" << "\033[1;33m" << " (" << 0 << '/' << m_total_steps << ')' << "\033[0m" << '\n';
  }


  /**
   * @brief Redraws the progress bar with a given percentage.
   * @param percentage The progress percentage to display
   */
  void redraw_bar(int percentage) {
    m_output << "\033[A\033[K\033[A\033[K\033[A\033[K\n"; // Clear previous output
    m_output << m_open_bracket;

    // Compute filled portion of the progress bar
    int completed = percentage / 2;

    m_output << "\033[32m"; // Green color for completed part
    for (int i = 0; i < completed; ++i) m_output << m_done_char;
    m_output << "\033[0m"; // Reset color

    for (int i = completed; i < 50; ++i) m_output << m_todo_char;
    
    m_output << m_close_bracket                                       // Close progress bar
            << "\033[1;39m"                                         // Bold, red text for percentage
            << ' ' << percentage << '%'                             // Print percentage
            << "\033[1;33m"                                         // Bold, yellow text for progress relative
            << " (" << m_current_step << '/' << m_total_steps << ')'  // Print progress relative to total iterations
            << "\033[0m"                                            // Reset text style
            << '\n';                                                // Move to next line
  
  }


  /**
   * @brief Prints the elapsed time since the progress bar started.
   */
  void print_elapsed_time() {
    auto now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - m_start_time);

    int hours = static_cast<int>(duration.count() / 3600);
    int minutes = static_cast<int>((duration.count() % 3600) / 60);
    int seconds = static_cast<int>(duration.count() % 60);

    m_output << "\033[1;31m" << "Elapsed time: " << "\033[33m"
            << std::setw(2) << std::setfill('0') << hours << ':'
            << std::setw(2) << std::setfill('0') << minutes << ':'
            << std::setw(2) << std::setfill('0') << seconds
            << "\033[0m" << '\n';
  }
};

#endif // PROGRESSBAR_HPP