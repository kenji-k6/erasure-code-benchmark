/**
 * @file bm_config.hpp
 * @brief Defines the BenchmarkConfig struct and related constants.
 */

#ifndef BM_CONFIG_HPP
#define BM_CONFIG_HPP

#define KiB *1024
#define MiB *1024*1024

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
  size_t message_size;                            ///< Size of the original message to be encoded (in bytes)
  size_t block_size;                              ///< Size of each block (in bytes)
  ECTuple ec_params;                            
  size_t num_lost_blocks;                         ///< Number of total blocks lost (recovery + original)

  size_t num_cpu_threads;                         ///< Number of CPU threads to use

  int num_iterations;                             ///< Number of iterations to run the benchmark
  int num_warmup_iterations;                      ///< Number of warm-up iterations

  XorecVersion xorec_version = XorecVersion::Scalar; ///< Version of XOR-EC to use (default: Scalar)

  bool gpu_computation;                           ///< Flag for GPU computation
  size_t num_gpu_blocks = 0;
  size_t threads_per_gpu_block = 0;

  ConsoleReporter* reporter = nullptr;
};


/// @typedef BenchmarkFunction
/// @brief Type definition for benchmark functions
using BenchmarkFunction = void(*)(benchmark::State&, const BenchmarkConfig&);

/// @brief Constants for fixed values
constexpr size_t MESSAGE_SIZE = 128 MiB;
constexpr size_t NUM_GPU_BLOCKS = 256;
constexpr size_t NUM_THREADS_PER_BLOCK = 512;
extern const std::vector<size_t> VAR_BLOCK_SIZES;
extern const std::vector<ECTuple> VAR_EC_PARAMS;
extern const std::vector<size_t> VAR_NUM_CPU_THREADS;
extern const std::vector<size_t> VAR_NUM_LOST_BLOCKS;
#endif // BM_CONFIG_HPP