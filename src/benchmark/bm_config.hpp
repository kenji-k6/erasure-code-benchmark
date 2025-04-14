/**
 * @file bm_config.hpp
 * @brief Defines the BenchmarkConfig struct and related constants.
 */

#ifndef BM_CONFIG_HPP
#define BM_CONFIG_HPP

#include "console_reporter.hpp"
#include "xorec_utils.hpp"
#include <cstdint>
#include <vector>

/**
 * @struct BenchmarkConfig
 * 
 * @brief Struct to hold the configuration for a single benchmark run
 */
struct BenchmarkConfig {
  size_t data_size;                               ///< Total size of original data (in bytes)
  size_t block_size;                              ///< Size of each block (in bytes)
  size_t num_lost_blocks;                         ///< Number of total blocks lost (recovery + original)
  double redundancy_ratio;                        ///< Recovery blocks / original blocks ratio
  int num_iterations;                             ///< Number of iterations to run the benchmark
  uint8_t plot_id;                                ///< Identifier for plotting

  size_t num_data_blocks;                   ///< Number of original data blocks
  size_t num_parity_blocks;                   ///< Number of recovery blocks

  bool is_xorec_config;                           ///< Flag to indicate whether this configuration is for XOR-EC algorithm(s)
  struct {
    XorecVersion version;
    bool unified_mem;
  } xorec_params;

  ConsoleReporter* reporter = nullptr;
};


/// @typedef BenchmarkFunction
/// @brief Type definition for benchmark functions
using BenchmarkFunction = void(*)(benchmark::State&, const BenchmarkConfig&);

/// @brief Constants for fixed values
constexpr int FIXED_GPU_BLOCKS = 8;
constexpr int FIXED_GPU_THREADS_PER_BLOCK = 32;
constexpr size_t FIXED_NUM_ORIGINAL_BLOCKS = 128;
constexpr size_t FIXED_NUM_RECOVERY_BLOCKS = 4;
constexpr size_t FIXED_BUFFER_SIZE = 1048576; // 1 MiB
constexpr double FIXED_PARITY_RATIO = 0.03125;
constexpr size_t FIXED_NUM_LOST_BLOCKS = 1;
extern const std::vector<size_t> VAR_BUFFER_SIZE;
extern const std::vector<size_t> VAR_NUM_RECOVERY_BLOCKS;
extern const std::vector<size_t> VAR_NUM_LOST_BLOCKS;

#endif // BM_CONFIG_HPP