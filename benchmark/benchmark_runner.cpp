#include "leopard.h"
#include "cm256.h"
#include "wirehair/wirehair.h"
#include "aff3ct.hpp"
#include "erasure_code.h"

#include <iostream>
#include <chrono>

void benchmark_aff3ct() {
  std::cout << "aff3ct Compiled!" << '\n';
}


void benchmark_cm256() {
  if (cm256_init()) {
    exit(1);
  }
  std::cout << "cm256 Compiled!" << '\n';
}


void benchmark_leopard() {
  if (leo_init()) {
    exit(1);
  }
  std::cout << "Leopard Compiled!" << '\n';
}


void benchmark_wirehair() {
  const WirehairResult initResult = wirehair_init();
  if (initResult != Wirehair_Success) {
    exit(1);
  }
  std::cout << "Wirehair Compiled!" << '\n';
}




int main() {
  benchmark_leopard();
  benchmark_cm256();
  benchmark_wirehair();
  benchmark_aff3ct();
  std::cout << "All libraries compiled!" << '\n';

  return 0;
}