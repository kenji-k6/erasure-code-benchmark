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

/**
 * @brief Parse command-line arguments and set global variables accordingly
 * 
 * @param argc 
 * @param argv 
 */
void parse_args(int argc, char** argv);

/**
 * @brief Populate the benchmark configurations based on the parsed command-line arguments
 * 
 * @param configs Vector to store the benchmark configurations (must be empty)
 * @param lost_block_idxs Vector to store the lost block indices for each configuration (must be empty)
 */
void get_configs(std::vector<BenchmarkConfig>& configs, std::vector<std::vector<uint32_t>>& lost_block_idxs);

/**
 * @brief Run the benchmarks based on the provided configurations
 * 
 * @attention Must be called after `get_configs`
 * 
 * @param configs Vector of benchmark configurations
 */
void run_benchmarks(std::vector<BenchmarkConfig>& configs);

#endif // BENCHMARK_UTILS_HPP