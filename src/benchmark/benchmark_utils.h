/**
 * @file benchmark_utils.h
 * @brief Utility functions and constants for parsing and validating command-line arguments
 */

#ifndef BENCHMARK_UTILS_H
#define BENCHMARK_UTILS_H

#include "benchmark_config.h"
#include "benchmark_reporters.h"
#include <string>
#include <unordered_set>
#include <unordered_map>

// Function declarations
void get_configs(int argc, char** argv, std::vector<BenchmarkConfig>& configs, std::vector<uint32_t>& lost_block_idxs);
void run_benchmarks(std::vector<BenchmarkConfig>& configs);

#endif // BENCHMARK_UTILS_H