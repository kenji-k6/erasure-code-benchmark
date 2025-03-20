/**
 * @file bm_utils.h
 * @brief Utility functions and constants for parsing and validating command-line arguments
 */

#ifndef BM_UTILS_HPP
#define BM_UTILS_HPP

#include "bm_config.hpp"
#include "bm_reporters.hpp"
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
 * @brief Populate the EC-algorithm benchmark configurations based on the parsed command-line arguments
 * 
 * @param ec_configs Vector to store the EC benchmark configurations (must be empty)
 * @param lost_block_idxs Vector to store the lost block indices for each EC configuration (must be empty)
 * @param ec_configs Vector to store the performance benchmark configurations (must be empty)
 */
void get_configs(std::vector<BenchmarkConfig>& ec_configs, std::vector<std::vector<uint32_t>>& lost_block_idxs, std::vector<BenchmarkConfig>& perf_configs);


/**
 * @brief Run the EC-algorithm benchmarks based on the provided configurations
 * 
 * @attention Must be called after `get_configs`
 * 
 * @param ec_configs Vector of EC benchmark configurations
 * @param perf_configs Vector of performance benchmark configurations
 */
void run_benchmarks(std::vector<BenchmarkConfig>& ec_configs, std::vector<BenchmarkConfig>& perf_configs);

#endif // BM_UTILS_HPP