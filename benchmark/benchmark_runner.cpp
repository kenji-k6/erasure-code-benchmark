#include "benchmark_runner.h"




int benchmark_leopard() {
  if (leo_init()) {
    // error in initialization
    return 1;
  }

  return 0;
}


int main() {
  std::cout << "All libraries compiled!" << '\n';

  return 0;
}