#include <iostream>
#include <chrono>
#include "leopard.h"


void benchmark_leopard() {
  // Original and recovery data must not exceed 65536 pieces
  // recovery_count <= original_count
  // buffer_bytes has to be a multiple of 64
  // each buffer should have same no. of bytes
  // even the last piece must be rounded up to the block size

  leo_init();
}


int main() {
  benchmark_leopard();
  return 0;
}