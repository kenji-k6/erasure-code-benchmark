#ifndef WIREHAIR_BENCHMARK_H
#define WIREHAIR_BENCHMARK_H
#include "abstract_benchmark.h"
#include "wirehair/wirehair.h"
#include "utils.h"

#include <iostream>
#include <vector>
#include <memory>
#include <cstring>

class WirehairBenchmark : public ECCBenchmark {
public:
  int setup() override;
  void teardown() override;
  int encode() override;
  int decode() override;
  void flush_cache() override;
  void check_for_corruption() override;
  void simulate_data_loss() override;

private:
  uint8_t* original_data_;
  WirehairCodec encoder_;
  WirehairCodec decoder_;
}; // class WirehairBenchmark

#endif // WIREHAIR_BENCHMARK_H
