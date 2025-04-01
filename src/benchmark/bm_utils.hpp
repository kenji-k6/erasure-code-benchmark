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

void run_benchmarks(int argc, char** argv);
#endif // BM_UTILS_HPP