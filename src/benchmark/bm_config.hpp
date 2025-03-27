/**
 * @file bm_config.hpp
 * @brief Defines the BenchmarkConfig struct and related constants.
 */

 #ifndef BM_CONFIG_HPP
 #define BM_CONFIG_HPP
 
 #include "bm_reporters.hpp"
 #include "xorec_utils.hpp"
 #include <cstdint>
 #include <vector>

 
using FECTuple = std::tuple<size_t, size_t>;

struct BenchmarkConfig {
  size_t message_size;                          ///< Size of the message to be encoded
  size_t block_size;                            ///< Size of the block to be encoded
  FECTuple fec_params;                          ///< Tuple to hold FEC parameters

  size_t num_lost_rmda_packets;

  bool is_gpu_config;
  size_t num_gpu_blocks;
  size_t threads_per_gpu_block;

  int num_iterations;                           ///< Number of iterations to run the benchmark
  BenchmarkProgressReporter *progress_reporter = nullptr;
};

using BenchmarkFunction = void(*)(benchmark::State&, const BenchmarkConfig&);

constexpr size_t FIXED_MESSAGE_SIZE = 128 * 1024 * 1024; // 128 MiB
constexpr size_t FIXED_NUM_LOST_RDMA_PKTS = 0;
const std::vector<size_t> VAR_BLOCK_SIZES = { 4096, 8192, 16384, 32768, 65536, 131072, 262144 };
const std::vector<FECTuple> VAR_FEC_PARAMS = { {2,1}, {4,2}, {8,4}, {16,4}, {16,8}, {32, 8}, {32,4} };
const std::vector<size_t> VAR_NUM_GPU_BLOCKS = { 8, 16, 32, 64, 128, 256, 512, 1024, 2048 };
const std::vector<size_t> VAR_THREADS_PER_GPU_BLOCK = { 32, 64, 128, 256, 512, 1024 };

#endif // BM_CONFIG_HPP