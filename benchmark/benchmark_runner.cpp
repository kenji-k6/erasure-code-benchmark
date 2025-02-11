// #include "aff3ct.hpp"
#include "cm256.h"
#include "gnuradio/fec/encoder.h"
#include "gnuradio/fec/decoder.h"

#include "leopard.h"
#include "wirehair/wirehair.h"

#include <iostream>
#include <chrono>


void benchmark_leopard() {
  if (leo_init()) {
    // error if leo_init doesn't return 0
    exit(1);
  }
  std::cout << "Sucess compiling leopard!" << '\n';
}


void benchmark_cm256() {
  if (cm256_init()) {
    // error if cm256_init doesn't return 0
    exit(1);
  }
  std::cout << "Sucess compiling cm256!" << '\n';
}


void benchmark_wirehair() {
  const WirehairResult initResult = wirehair_init();
  if (initResult != Wirehair_Success) {
    exit(1);
  }
  std::cout << "Sucess compiling Wirehair!" << '\n';
}


void benchmark_gnuradio() {
  std::cout << "Todo: Compile GnuradioFEC" << '\n';
}


int main() {
  benchmark_leopard();
  benchmark_cm256();
  benchmark_wirehair();
  benchmark_gnuradio();

  return 0;
}