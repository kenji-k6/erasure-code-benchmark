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


using ECTuple = std::tuple<size_t, size_t>;

/**
 * @struct BenchmarkConfig
 * 
 * @brief Struct to hold the configuration for a single benchmark run
 */
struct BenchmarkConfig {
  size_t data_size;                               ///< Total size of original data (in bytes)
  size_t block_size;                              ///< Size of each block (in bytes)
  ECTuple ec_params;                            
  size_t num_lost_blocks;                         ///< Number of total blocks lost (recovery + original)
  int num_iterations;                             ///< Number of iterations to run the benchmark

  XorecVersion xorec_version;

  bool gpu_computation;                   ///< Flag for GPU computation
  size_t num_gpu_blocks;
  size_t threads_per_gpu_block;

  ConsoleReporter* reporter = nullptr;
};


/// @typedef BenchmarkFunction
/// @brief Type definition for benchmark functions
using BenchmarkFunction = void(*)(benchmark::State&, const BenchmarkConfig&);

/// @brief Constants for fixed values
extern const std::vector<size_t> VAR_BLOCK_SIZES;
extern const std::vector<ECTuple> VAR_EC_PARAMS;
extern const std::vector<size_t> VAR_NUM_LOST_BLOCKS;
extern const std::vector<size_t> VAR_NUM_GPU_BLOCKS;
extern const std::vector<size_t> VAR_NUM_THREADS_PER_BLOCK;

#endif // BM_CONFIG_HPP