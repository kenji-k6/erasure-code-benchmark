#include "bm_utils.hpp"


int main (int argc, char** argv) {
  parse_args(argc, argv);
  run_benchmarks();
  return 0;
}