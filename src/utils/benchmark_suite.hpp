/**
 * @file benchmark_suite.hpp
 * @brief Exposes the main run_benchmarks function.
 */

#ifndef BM_SUITE_HPP
#define BM_SUITE_HPP
#include "bm_config.hpp"

// Type definition for convenience
using BenchmarkTuple = std::tuple<std::string, BenchmarkFunction, BenchmarkConfig>;

/**
 * @brief Parse the command-line arguments, construct the benchmarks and run them
 * 
 * @param argc 
 * @param argv 
 */
void run_benchmarks(int argc, char** argv);
#endif // BM_SUITE_HPP