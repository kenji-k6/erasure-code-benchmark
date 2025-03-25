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

 

using BenchmarkFunction = void(*)(benchmark::State&, const BenchmarkConfig&);
using FECTuple = std::tuple<size_t, size_t>;

struct BenchmarkConfig {
  size_t message_size;                          ///< Size of the message to be encoded
  size_t block_size;                            ///< Size of the block to be encoded
  FECTuple fec_params;                          ///< Tuple to hold FEC parameters
  
  BenchmarkProgressReporter *progress_reporter = nullptr;
};
 

constexpr size_t FIXED_MESSAGE_SIZE = 128 * 1024 * 1024; // 128 MiB
const std::vector<size_t> VAR_BLOCK_SIZES = { 4096, 8192, 16384, 32768, 65536, 131072, 262144 };
const std::vector<FECTuple> VAR_FEC_PARAMS = { {2,1}, {4,2}, {8,4}, {16,4}, {16,8}, {32, 8}, {32,4} };
#endif // BM_CONFIG_HPP