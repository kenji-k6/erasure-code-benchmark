/**
 * @file benchmark_utils.h
 * @brief Utility functions and constants for parsing and validating command-line arguments
 */

#ifndef BENCHMARK_UTILS_HPP
#define BENCHMARK_UTILS_HPP

#include "benchmark_config.hpp"
#include "benchmark_reporters.hpp"
#include <unordered_set>
#include <unordered_map>

// Function declarations
void parse_args(int argc, char** argv);
void get_configs(std::vector<BenchmarkConfig>& configs, std::vector<std::vector<uint32_t>>& lost_block_idxs);
void run_benchmarks(std::vector<BenchmarkConfig>& configs);

#endif // BENCHMARK_UTILS_HPP