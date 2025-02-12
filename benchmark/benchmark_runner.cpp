#include "benchmark_runner.h"
#include "utils.h"



int benchmark_leopard() {
  if (leo_init()) {
    // error in initialization
    return 1;
  }

  return 0;
}


int main() {
  PCGRandom rng1(0, 0);
  PCGRandom rng2(0, 0);

  std::cout << rng1.next() << '\n' << rng2.next() << "\n\n";
  std::cout << rng1.next() << '\n' << rng2.next() << "\n\n";
  std::cout << rng1.next() << '\n' << rng2.next() << "\n\n";
  std::cout << rng1.next() << '\n' << rng2.next() << "\n\n";

  return 0;
}