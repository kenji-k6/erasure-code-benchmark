#ifndef PROGRESSBAR_H
#define PROGRESSBAR_H
#include <iostream>
#include <ostream>
#include <string>
#include <stdexcept>
#include <chrono>

class ProgressBar {
public:
  ~ProgressBar();

  ProgressBar (ProgressBar const&) = delete;
  ProgressBar& operator=(ProgressBar const&) = delete;
  ProgressBar (ProgressBar&&) = delete;
  ProgressBar& operator=(ProgressBar&&) = delete;

  inline ProgressBar(int n_iter, bool showbar=true, std:: ostream& out=std::cerr);

  inline void reset();
  inline void update();


private:
  int progress;
  int n_cycles;
  int last_perc;
  bool do_show_bar;
  bool update_is_called;

  std::string done_char;
  std::string todo_char;
  std::string opening_bracket_char;
  std::string closing_bracket_char;

  std::ostream& output;
};

inline ProgressBar::ProgressBar(int n_iter, bool showbar, std::ostream& out) :
  progress(0),
  n_cycles(n_iter),
  last_perc(0),
  do_show_bar(showbar),
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
    if (do_show_bar) {
      output << opening_bracket_char;
      for (int _ = 0; _ < 50; _++) output << todo_char;
      output << closing_bracket_char << " 0%";
    }
    else {
      output << "0%";
    }
  }

  update_is_called = true;

  // compute percentage, if did not change, do nothing and return
  int perc = 0;
  perc = progress*100.0/(n_cycles-1);
  if (perc < last_perc) return;

  // update percentage
  if (perc == last_perc + 1) {
    // erase the corret number of characters
    if (perc <= 10) output << "\b\b" << perc << '%';
    else if (perc > 10 and perc < 100) output << "\b\b\b" << perc << '%';
    else if (perc == 100) output << "\b\b\b" << perc << '%';
  }

  if (do_show_bar == true) {
    // update bar every ten units
    if (perc % 2 == 0) {
      // erase closing bracket
      output << std::string(closing_bracket_char.size(), '\b');
      //erase trailing percentage characters
      if (perc < 10) output << "\b\b\b";
      else if (perc >= 10 && perc < 100) output << "\b\b\b\b";
      else if (perc == 100) output << "\b\b\b\b\b";

      //erase 'todo_char'
      for (int j = 0; j < 50-(perc-1)/2; ++j) {
        output << std::string(todo_char.size(), '\b');
      }

      // add one additional 'done_char'
      if (perc == 0) output << todo_char;
      else output << done_char;

      // refill with 'todo_char'
      for (int j = 0; j < 50-(perc-1)/2-1; ++j) {
        output << todo_char;
      }

      //re-add trailing percentage characters
      output << closing_bracket_char << ' ' << perc << '%';
    }
  }

  last_perc = perc;
  ++progress;
  output << std::flush;
  return;
}
#endif // PROGRESSBAR_H