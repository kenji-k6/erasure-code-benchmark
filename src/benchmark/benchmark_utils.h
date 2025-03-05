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

constexpr const char* OUTPUT_FILE_DIR = "../results/raw/";
extern const std::unordered_map<std::string, BenchmarkFunction> available_benchmarks;

// Function declarations
void get_configs(int argc, char** argv, std::vector<BenchmarkConfig>& configs, std::vector<uint32_t>& lost_block_idxs);
std::string get_filename();
void register_benchmarks(std::vector<BenchmarkConfig>& configs, BenchmarkProgressReporter *console_reporter);
void run_benchmarks(std::vector<BenchmarkConfig>& configs, BenchmarkProgressReporter *console_reporter, BenchmarkCSVReporter *csv_reporter);

static void usage();
static void check_args(uint64_t s, uint64_t b, uint32_t l, double r, int i, uint32_t num_orig_blocks, uint32_t num_rec_blocks);
static void get_full_benchmark_configs(int num_iterations, std::vector<BenchmarkConfig>& configs, std::vector<uint32_t>& lost_block_idxs);
static BenchmarkConfig get_single_benchmark_config(uint64_t s, uint64_t b, uint32_t l, double r, int i, std::vector<BenchmarkConfig>& configs, std::vector<uint32_t>& lost_block_idxs);

// Erasure Code (EC) constraints
namespace ECLimits {
  constexpr size_t BASELINE_BLOCK_ALIGNMENT = 64;

  constexpr size_t CM256_MAX_TOT_BLOCKS = 256;

  constexpr size_t ISAL_MIN_BLOCK_SIZE = 64;
  constexpr size_t ISAL_MAX_DATA_BLOCKS = 256;
  constexpr size_t ISAL_MAX_TOT_BLOCKS = 256;

  constexpr size_t LEOPARD_MAX_TOT_BLOCKS = 65'536;
  constexpr size_t LEOPARD_BLOCK_ALIGNMENT = 64;

  constexpr size_t WIREHAIR_MIN_DATA_BLOCKS = 2;
  constexpr size_t WIREHAIR_MAX_DATA_BLOCKS = 64'000;
}

#endif // BENCHMARK_UTILS_H