/**
 * @file benchmark_config.h
 * @brief Defines the BenchmarkConfig struct and related constants.
 */


#ifndef BENCHMARK_CONFIG_H
#define BENCHMARK_CONFIG_H

#include "benchmark_reporters.h"
#include <cstdint>
#include <vector>

/// @brief Configuration for a benchmark run
struct BenchmarkConfig {
  uint64_t data_size;               ///< Total size of original data
  uint64_t block_size;              ///< Size of each block
  uint64_t num_lost_blocks;         ///< Number of total blocks lost (recovery + original)
  double redundancy_ratio;          ///< Recovery blocks / original blocks ratio
  int num_iterations;               ///< Number of iterations to run the benchmark
  uint8_t plot_id;                  ///< Identifier for plotting
  const uint32_t *lost_block_idxs;  ///< Pointer to the lost block indices array

  struct {
    uint32_t num_original_blocks;   ///< Number of original data blocks
    uint32_t num_recovery_blocks;   ///< Number of recovery blocks
  } computed;

  BenchmarkProgressReporter *progress_reporter = nullptr;
};

/// @Brief Alias for benchmark function type
using BenchmarkFunction = void(*)(benchmark::State&, const BenchmarkConfig&);
using BMTuple = std::tuple<std::string, BenchmarkFunction>;

/// Constants for benchmark configurations (when running full benchmark)
constexpr uint32_t FIXED_NUM_ORIGINAL_BLOCKS = 128;
constexpr uint32_t FIXED_NUM_RECOVERY_BLOCKS = 4;
constexpr uint64_t FIXED_BUFFFER_SIZE = 4194304; // 4 MiB
constexpr double FIXED_PARITY_RATIO = 0.03125;
constexpr uint64_t FIXED_NUM_LOST_BLOCKS = 1;
const std::vector<uint64_t> VAR_BUFFER_SIZE = { 134217728, 67108864, 33554432, 16777216, 8388608, 4194304, 2097152, 1048576, 524288, 262144 };
const std::vector<uint32_t> VAR_NUM_RECOVERY_BLOCKS = { 128, 64, 32, 16, 8, 4, 2, 1 };
const std::vector<uint32_t> VAR_NUM_LOST_BLOCKS = { 128, 64, 32, 16, 8, 4, 2, 1 };

#endif // BENCHMARK_CONFIG_H