/**
 * @file benchmark_utils.cpp
 * @brief Implementations of utility functions for parsing and validating command-line arguments
 */


#include "bm_cli.hpp"
#include "bm_utils.hpp"
#include "benchmark/benchmark.h"
#include "bm_functions.hpp"
#include "utils.hpp"
#include <filesystem>
#include <getopt.h>
#include <ranges>
#include <unordered_map>
#include <unordered_set>


// Global variable definitions
constexpr const char* RAW_DIR = "../results/raw/";
std::string RESULT_DIR = "";
std::string EC_OUTPUT_FILE_NAME = "ec_results.csv";
std::string PERF_OUTPUT_FILE_NAME = "perf_results.csv";

bool OVERWRITE_FILE = true;
int NUM_ITERATIONS = 10;

int NUM_BASE_CONFIGS              = 0;
int NUM_XOREC_CPU_CONFIGS         = 0;
int NUM_XOREC_UNIFIED_PTR_CONFIGS = 0;
int NUM_XOREC_GPU_PTR_CONFIGS     = 0;
int NUM_XOREC_GPU_CMP_CONFIGS     = 0;

bool RUN_XOREC_SCALAR = false;
bool RUN_XOREC_AVX    = false;
bool RUN_XOREC_AVX2   = false;
bool RUN_XOREC_AVX512 = false;

bool RUN_PERF_BM  = false;
bool RUN_EC_BM    = false;

bool INIT_BASE_CONFIGS              = false;
bool INIT_XOREC_CPU_CONFIGS         = false;
bool INIT_XOREC_UNIFIED_PTR_CONFIGS = false;
bool INIT_XOREC_GPU_PTR_CONFIGS     = false;

bool RUN_XOREC_NO_PREFETCH  = false;
bool RUN_XOREC_PREFETCH     = false;

bool RUN_XOREC_TOUCH_UNIFIED    = false;
bool RUN_XOREC_NO_TOUCH_UNIFIED = false;

std::chrono::system_clock::time_point START_TIME;
using BenchmarkTuple = std::tuple<std::string, BenchmarkFunction, BenchmarkConfig>;

std::unordered_set<std::string> selected_base_ec_benchmarks;
std::unordered_set<std::string> selected_xorec_ec_benchmarks;
std::unordered_set<std::string> selected_perf_benchmarks;

const std::unordered_map<std::string, BenchmarkFunction> available_base_ec_benchmarks = {
  { "cm256",                  BM_CM256                },
  { "isal",                   BM_ISAL                 },
  { "leopard",                BM_Leopard              },
  { "wirehair",               BM_Wirehair             }
};

const std::unordered_map<std::string, BenchmarkFunction> available_xorec_ec_benchmarks = {
  { "xorec-cpu",              BM_XOREC              },
  { "xorec-unified-ptr",      BM_XOREC_UNIFIED_PTR  },
  { "xorec-gpu-ptr",          BM_XOREC_GPU_PTR      },
  { "xorec-gpu-cmp",          BM_XOREC_GPU_CMP      }
};

const std::unordered_map<std::string, BenchmarkFunction> available_perf_benchmarks = {
  { "perf-xorec-scalar", BM_XOR_BLOCKS_SCALAR },
  { "perf-xorec-avx",    BM_XOR_BLOCKS_AVX    },
  { "perf-xorec-avx2",   BM_XOR_BLOCKS_AVX2   },
  { "perf-xorec-avx512", BM_XOR_BLOCKS_AVX512 }
};

const std::unordered_map<std::string, std::string> ec_benchmark_names = {
  { "cm256",                "CM256"                 },
  { "isal",                 "ISA-L"                 },
  { "leopard",              "Leopard"               },
  { "wirehair",             "Wirehair"              },
  { "xorec-cpu",            "XOR-EC (CPU)"          },
  { "xorec-unified-ptr",    "XOR-EC (Unified Ptr)"  },
  { "xorec-gpu-ptr",        "XOR-EC (GPU Ptr)"      },
  { "xorec-gpu-cmp",        "XOR-EC (GPU Cmp)"      }
};

const std::unordered_map<std::string, std::string> perf_benchmark_names = {
  { "perf-xorec-scalar", "Theoretical XOR-EC (SCALAR)" },
  { "perf-xorec-avx",    "Theoretical XOR-EC (AVX)"    },
  { "perf-xorec-avx2",   "Theoretical XOR-EC (AVX2)"   },
  { "perf-xorec-avx512", "Theoretical XOR-EC (AVX512)" }
};

const std::unordered_map<std::string, XorecVersion> perf_benchmark_version = {
  { "perf-xorec-scalar", XorecVersion::Scalar },
  { "perf-xorec-avx",    XorecVersion::AVX   },
  { "perf-xorec-avx2",   XorecVersion::AVX2   },
  { "perf-xorec-avx512", XorecVersion::AVX512 }
};

/**
 * @brief Prints usage information and exits the program
 */
static void usage() {
  print_usage();
  print_options();
  exit(0);
}

/**
 * @brief Adds a benchmark to the corresponding `selected_*_benchmarks` set based on its input name.
 * 
 * @param name The name of the benchmark to add
 */
static void inline add_benchmark(std::string name) {
  if (available_base_ec_benchmarks.find(name) != available_base_ec_benchmarks.end()) {
    selected_base_ec_benchmarks.insert(name);
  } else if (available_xorec_ec_benchmarks.find(name) != available_xorec_ec_benchmarks.end()) {
    selected_xorec_ec_benchmarks.insert(name);
  } else if (available_perf_benchmarks.find(name) != available_perf_benchmarks.end()) {
    selected_perf_benchmarks.insert(name);
  } else {
    throw_error("Invalid benchmark: "+name);
  }
}


/**
 * @brief Splits a comma-seperated argument string into a vector of strings
 * 
 * @param input The input string to split
 * @return A vector of strings
 */
std::vector<std::string> get_arg_vector(std::string input) {
  std::vector<std::string> result;
  size_t start = 0;
  size_t end;
  while ((end = input.find(",", start)) != std::string::npos) {
    result.push_back(to_lower(input.substr(start, end-start)));
    start = end + 1;
  }
  result.push_back(to_lower(input.substr(start)));
  return result;
}


/**
 * @brief Parses command-line arguments and sets the global configuration variables accordingly.
 * 
 * @param argc 
 * @param argv 
 */
void parse_args(int argc, char** argv) {
  struct option long_options[] = {
    { "help",           no_argument,        nullptr, 'h'  },
    { "result-dir",     required_argument,  nullptr, 'r'  },
    { "append",         no_argument,        nullptr, 'a'  },
    { "benchmark",      required_argument,  nullptr, 'b'  },
    { "iterations",     required_argument,  nullptr, 'i'  },
    { "base",           required_argument,  nullptr,  0   },
    { "xorec",          required_argument,  nullptr,  0   },
    { "simd",           required_argument,  nullptr,  0   },
    { "touch-unified",  required_argument,  nullptr,  0   },
    { "prefetch",       required_argument,  nullptr,  0   },
    { "perf-xorec",     required_argument,  nullptr,  0   },
    { nullptr,          0,                  nullptr,  0   }
  };

  int c;
  int option_index = 0;
  std::string flag;
  std::vector<std::string> args;

  while ((c = getopt_long(argc, argv, "hs:b:l:r:i:", long_options, &option_index)) != -1) {
    switch (c) {
      case 'h':
        usage();
        break;
      case 'i':
        NUM_ITERATIONS = std::stoi(optarg);
        break;
      case 'r':
        RESULT_DIR = std::string(optarg) + "/";
        break;
      case 'a':
        OVERWRITE_FILE = false;
        break;
      case 'b':
        args = get_arg_vector(std::string(optarg));
        for (auto arg : args) {
          if (arg == "ec") {
            RUN_EC_BM = true;
          } else if (arg == "perf") {
            RUN_PERF_BM = true;
          } else {
            std::cerr << "Error: --benchmark option must be either 'ec', 'perf', or 'all'.\n";
            exit(0);
          }
        }
        break;
      case 0:
        flag = std::string(long_options[option_index].name);
        args = get_arg_vector(std::string(optarg));
        
        if (flag == "base") {
          for (auto arg : args) {
            add_benchmark(arg);
          }
        } else if (flag == "xorec") {
          for (auto arg : args) {
            add_benchmark("xorec-"+arg);
          }
        } else if (flag == "simd") {
          for (auto arg : args) {
            if (arg == "scalar") {
             RUN_XOREC_SCALAR = true;
            } else if (arg == "avx") {
              RUN_XOREC_AVX = true;
            } else if (arg == "avx2") {
              RUN_XOREC_AVX2 = true;
            } else if (arg == "avx512") {
              RUN_XOREC_AVX512 = true;
            } else {
              std::cerr << "Error: --simd option must be either 'scalar', 'avx', 'avx2', or 'avx512'.\n";
              exit(0);
            }
          }
        } else if (flag == "touch-unified") {
          for (auto arg : args) {
            if (arg == "true" || arg == "1") {
              RUN_XOREC_TOUCH_UNIFIED = true;
            } else if (arg == "false" || arg == "0") {
              RUN_XOREC_NO_TOUCH_UNIFIED = true;
            } else {
              std::cerr << "Error: --touch-unified option must be either 'true' or 'false'.\n";
              exit(0);
            }
          }
        } else if (flag == "prefetch") {
          for (auto arg : args) {
            if (arg == "true" || arg == "1") {
              RUN_XOREC_PREFETCH = true;
            } else if (arg == "false" || arg == "0") {
              RUN_XOREC_NO_PREFETCH = true;
            } else {
              std::cerr << "Error: --prefetch option must be either 'true' or 'false'.\n";
              exit(0);
            }
          }
        } else if (flag == "perf-xorec") {
          for (auto arg : args) {
            add_benchmark("perf-xorec-"+arg);
          }
        } else {
          std::cerr << "Error: Invalid option: " << flag << '\n';
          exit(0);
        }
        break;
      default:
        usage();
        exit(0);
    }
  }

  
  if (selected_base_ec_benchmarks.empty() && selected_xorec_ec_benchmarks.empty()) {
    for (const auto& [name, _] : available_base_ec_benchmarks) {
      add_benchmark(name);
    }
    
    for (const auto& [name, _] : available_xorec_ec_benchmarks) {
      add_benchmark(name);
    }
  }

  if (selected_perf_benchmarks.empty()) {
    for (const auto& [name, _] : available_perf_benchmarks) {
      add_benchmark(name);
    }
  }
  
  if (!RUN_XOREC_SCALAR && !RUN_XOREC_AVX && !RUN_XOREC_AVX2 && !RUN_XOREC_AVX512) {
    RUN_XOREC_SCALAR = true;
    RUN_XOREC_AVX = true;
    RUN_XOREC_AVX2 = true;
    RUN_XOREC_AVX512 = true;
  }

  if (!RUN_XOREC_TOUCH_UNIFIED && !RUN_XOREC_NO_TOUCH_UNIFIED) {
    RUN_XOREC_NO_TOUCH_UNIFIED = true;
  }

  if (!RUN_XOREC_PREFETCH && !RUN_XOREC_NO_PREFETCH) {
    RUN_XOREC_NO_PREFETCH = true;
  }

  if (!RUN_EC_BM && !RUN_PERF_BM) {
    RUN_EC_BM = true;
    RUN_PERF_BM = true;
  }
}

/**
 * @brief Initliazes the lost block indices used for benchmarking
 * 
 * @param lost_block_idxs A vector of vectors to store the lost block indices (should be empty)
 */
static void init_lost_block_idxs(std::vector<std::vector<uint32_t>>& lost_block_idxs) {
  for ([[maybe_unused]] auto _ : VAR_BUFFER_SIZE) {
    std::vector<uint32_t> vec;
    select_lost_block_idxs(
      FIXED_NUM_RECOVERY_BLOCKS,
      FIXED_NUM_LOST_BLOCKS,
      FIXED_NUM_ORIGINAL_BLOCKS + FIXED_NUM_RECOVERY_BLOCKS,
      vec
    );
    lost_block_idxs.push_back(vec);
  }

  for (auto num_rec_blocks : VAR_NUM_RECOVERY_BLOCKS) {
    std::vector<uint32_t> vec;
    select_lost_block_idxs(
      num_rec_blocks,
      FIXED_NUM_LOST_BLOCKS,
      FIXED_NUM_ORIGINAL_BLOCKS + num_rec_blocks,
      vec
    );
    lost_block_idxs.push_back(vec);
  }

  for (auto num_lost_blocks : VAR_NUM_LOST_BLOCKS) {
    std::vector<uint32_t> vec;
    select_lost_block_idxs(
      FIXED_NUM_ORIGINAL_BLOCKS,
      num_lost_blocks,
      FIXED_NUM_ORIGINAL_BLOCKS + FIXED_NUM_ORIGINAL_BLOCKS,
      vec
    );
    lost_block_idxs.push_back(vec);
  }
}

/**
 * @brief Get the XOR-EC GPU computation configs based on already initialized configs
 * 
 * @attention This function assumes that the XOR-EC GPU Pointer configurations have already been initialized
 * and are stored in the configs vector already
 * 
 * @param configs 
 */
static void get_xorec_gpu_cmp_configs(std::vector<BenchmarkConfig>& configs) {
  if (!INIT_XOREC_GPU_PTR_CONFIGS) throw_error("XOR-EC GPU Pointer configurations must be initialized before XOR-EC GPU Computation configurations.");
  
  for (int i = 0; i < NUM_BASE_CONFIGS; ++i) {
    BenchmarkConfig config = configs[i];
    config.is_xorec_config = true;
    config.xorec_params.gpu_mem = true;
    config.xorec_params.gpu_cmp = true;

    configs.push_back(config);
    ++NUM_XOREC_GPU_CMP_CONFIGS;
  }
}

/**
 * @brief Get the XOR-EC GPU Pointer configs based on already initialized configs
 * 
 * @attention This function assumes that the XOR-EC Unified Pointer configurations have already been initialized
 * and are stored in the configs vector already
 * 
 * @param configs 
 */
static void get_xorec_gpu_ptr_configs(std::vector<BenchmarkConfig>& configs) {
  if (!INIT_XOREC_UNIFIED_PTR_CONFIGS) throw_error("XOR-EC CPU configurations must be initialized before XOR-EC GPU Pointer configurations.");

  for (int i = NUM_BASE_CONFIGS; i < NUM_BASE_CONFIGS+NUM_XOREC_CPU_CONFIGS; ++i) {
    BenchmarkConfig config = configs[i];
    config.xorec_params.gpu_mem = true;
    configs.push_back(config);
    ++NUM_XOREC_GPU_PTR_CONFIGS;
  }

  INIT_XOREC_GPU_PTR_CONFIGS = true;
}

/**
 * @brief Get the XOR-EC Unified Pointer configs based on already initialized configs
 * 
 * @attention This function assumes that the XOR-EC CPU configurations have already been initialized
 * and are stored in the configs vector already
 * 
 * @param configs 
 */
static void get_xorec_unified_ptr_configs(std::vector<BenchmarkConfig>& configs) {
  if (!INIT_XOREC_CPU_CONFIGS) throw_error("XOR-EC CPU configurations must be initialized before XOR-EC GPU Pointer configurations.");

  std::vector<BenchmarkConfig> prefetch_configs;

  for (int i = NUM_BASE_CONFIGS; i < NUM_BASE_CONFIGS+NUM_XOREC_CPU_CONFIGS; ++i) {
    BenchmarkConfig config = configs[i];
    config.xorec_params.unified_mem = true;

    if (RUN_XOREC_PREFETCH) {
      config.xorec_params.prefetch = true;
      prefetch_configs.push_back(config);
    }

    if (RUN_XOREC_NO_PREFETCH) {
      config.xorec_params.prefetch = false;
      prefetch_configs.push_back(config);
    }
  }

  for (BenchmarkConfig config : prefetch_configs) {
    if (RUN_XOREC_TOUCH_UNIFIED) {
      config.xorec_params.touch_unified_mem = true;
      configs.push_back(config);
      ++NUM_XOREC_UNIFIED_PTR_CONFIGS;
    }

    if (RUN_XOREC_NO_TOUCH_UNIFIED) {
      config.xorec_params.touch_unified_mem = false;
      configs.push_back(config);
      ++NUM_XOREC_UNIFIED_PTR_CONFIGS;
    }
  }
  
  INIT_XOREC_UNIFIED_PTR_CONFIGS = true;
}

/**
 * @brief Get the XOR-EC CPU configs based on already initialized configs
 * 
 * @attention This function assumes that the base configurations have already been initialized
 * and are stored in the configs vector already
 * 
 * @param configs 
 */
static void get_xorec_cpu_configs(std::vector<BenchmarkConfig>& configs) {
  if (!INIT_BASE_CONFIGS) throw_error("Base configurations must be initialized before XOR-EC CPU configurations.");

  for (int i = 0; i < NUM_BASE_CONFIGS; ++i) {
    BenchmarkConfig config = configs[i];
    config.is_xorec_config = true;
    
    if (RUN_XOREC_SCALAR) {
      config.xorec_params.version = XorecVersion::Scalar;
      configs.push_back(config);
      ++NUM_XOREC_CPU_CONFIGS;
    }
    if (RUN_XOREC_AVX) {
      config.xorec_params.version = XorecVersion::AVX;
      configs.push_back(config);
      ++NUM_XOREC_CPU_CONFIGS;
    }
    if (RUN_XOREC_AVX2) {
      config.xorec_params.version = XorecVersion::AVX2;
      configs.push_back(config);
      ++NUM_XOREC_CPU_CONFIGS;
    }
    if (RUN_XOREC_AVX512) {
      config.xorec_params.version = XorecVersion::AVX512;
      configs.push_back(config);
      ++NUM_XOREC_CPU_CONFIGS;
    }
  }
  INIT_XOREC_CPU_CONFIGS = true;
}

/**
 * @brief Get the base EC configs based on already initialized configs
 * 
 * @param configs (should be empty)
 */
static void get_base_configs(std::vector<BenchmarkConfig>& configs, std::vector<std::vector<uint32_t>>& lost_block_idxs) {
  auto it = lost_block_idxs.begin();
  for (auto buf_size : VAR_BUFFER_SIZE) {
    BenchmarkConfig config {
      buf_size,
      buf_size / FIXED_NUM_ORIGINAL_BLOCKS,
      FIXED_NUM_LOST_BLOCKS,
      FIXED_PARITY_RATIO,
      NUM_ITERATIONS,
      0,
      *(it++),
      FIXED_NUM_ORIGINAL_BLOCKS,
      FIXED_NUM_RECOVERY_BLOCKS,
      false,
      { XorecVersion::Scalar, false, false, false, false, false },
      nullptr
    };
    configs.push_back(config);
    ++NUM_BASE_CONFIGS;
  }

  for (auto num_rec_blocks : VAR_NUM_RECOVERY_BLOCKS) {
    BenchmarkConfig config {
      FIXED_BUFFER_SIZE,
      FIXED_BUFFER_SIZE / FIXED_NUM_ORIGINAL_BLOCKS,
      FIXED_NUM_LOST_BLOCKS,
      static_cast<double>(num_rec_blocks) / FIXED_NUM_ORIGINAL_BLOCKS,
      NUM_ITERATIONS,
      1,
      *(it++),
      FIXED_NUM_ORIGINAL_BLOCKS,
      num_rec_blocks,
      false,
      { XorecVersion::Scalar, false, false, false, false, false },
      nullptr
    };
    configs.push_back(config);
    ++NUM_BASE_CONFIGS;
  }

  for (auto num_lost_blocks : VAR_NUM_LOST_BLOCKS) {
    BenchmarkConfig config {
      FIXED_BUFFER_SIZE,
      FIXED_BUFFER_SIZE / FIXED_NUM_ORIGINAL_BLOCKS,
      num_lost_blocks,
      1.0,
      NUM_ITERATIONS,
      2,
      *(it++),
      FIXED_NUM_ORIGINAL_BLOCKS,
      FIXED_NUM_ORIGINAL_BLOCKS,
      false,
      { XorecVersion::Scalar, false, false, false, false, false },
      nullptr
    };
    configs.push_back(config);
    ++NUM_BASE_CONFIGS;
  }

  INIT_BASE_CONFIGS = true;
}

/**
 * @brief Populate the EC benchmark configurations based on the user-passed arguments
 * 
 * @param configs Should be empty
 * @param lost_block_idxs Should be empty
 */
static void get_ec_configs(std::vector<BenchmarkConfig>& configs, std::vector<std::vector<uint32_t>>& lost_block_idxs) {
  init_lost_block_idxs(lost_block_idxs);

  get_base_configs(configs, lost_block_idxs);

  if (selected_xorec_ec_benchmarks.size() > 0) {
    get_xorec_cpu_configs(configs);
    get_xorec_unified_ptr_configs(configs);
    get_xorec_gpu_ptr_configs(configs);
    get_xorec_gpu_cmp_configs(configs);
  }
}

/**
 * @brief Gets the configs for the performance benchmarks
 * 
 * @param configs Should be empty
 */
static void get_perf_configs(std::vector<BenchmarkConfig>& configs) {
  for (auto data_size : VAR_BUFFER_SIZE) {
    size_t block_size = data_size / FIXED_NUM_ORIGINAL_BLOCKS;
    BenchmarkConfig config {
      0,
      block_size,
      0,
      0.0,
      0,
      0xFF,
      {},
      FIXED_NUM_ORIGINAL_BLOCKS,
      FIXED_NUM_RECOVERY_BLOCKS,
      true,
      { XorecVersion::Scalar, false, false, false, false, false },
      nullptr
    };
    configs.push_back(config);
  }
}

/**
 * @brief Get the benchmark name based on the algorithm and config parameters
 * 
 * @param inp_name 
 * @param config 
 * @return std::string 
 */
static std::string get_ec_benchmark_name(const std::string inp_name, BenchmarkConfig config) {
  std::string name = ec_benchmark_names.at(inp_name);
  if (config.is_xorec_config && !config.xorec_params.gpu_cmp) {
    name += ", " + get_version_name(config.xorec_params.version);

    if (config.xorec_params.unified_mem) {
      if (config.xorec_params.touch_unified_mem) {
        name += ",\nTouched";
      }

      if (config.xorec_params.prefetch) {
        name += ", prefetched";
      } else {
        name += ", on demand";
      }

    } else if (config.xorec_params.gpu_mem) {
      name += ",\nprefetched";
    }
  }

  return name;
}

/**
 * @brief Get the benchmarking function based on the algorithm name
 * 
 * @param name 
 * @return BenchmarkFunction 
 */
static BenchmarkFunction get_ec_benchmark_func(const std::string& name) {
  if (available_base_ec_benchmarks.find(name) != available_base_ec_benchmarks.end()) {
    return available_base_ec_benchmarks.at(name);
  } else {
    return available_xorec_ec_benchmarks.at(name);
  }
}

/**
 * @brief Get the performance benchmark name based on the input name
 * 
 * @param inp_name 
 * @return std::string 
 */
static std::string get_perf_benchmark_name(const std::string inp_name) {
  return perf_benchmark_names.at(inp_name);
}

/**
 * @brief Get the performance benchmark function based on the input name
 * 
 * @param name 
 * @return BenchmarkFunction 
 */
static BenchmarkFunction get_perf_benchmark_func(const std::string& name) {
  return available_perf_benchmarks.at(name);
}

/**
 * @brief "Zips" the EC benchmark names, functions, and configurations into a vector of tuples
 * 
 * @param configs 
 * @return std::vector<BenchmarkTuple> 
 */
static std::vector<BenchmarkTuple> get_ec_benchmarks(std::vector<BenchmarkConfig>& configs) {
  std::vector<BenchmarkTuple> benchmarks;

  auto it = configs.begin();
  // Base benchmarks
  for (; it != configs.begin()+NUM_BASE_CONFIGS; ++it) {
    for (auto& inp_name : selected_base_ec_benchmarks) {
      auto bm_name = get_ec_benchmark_name(inp_name, *it);
      auto bm_func = get_ec_benchmark_func(inp_name);
      benchmarks.push_back({ bm_name, bm_func, *it });
    }
  }

  // XOR-EC CPU benchmarks
  for (; it != configs.begin()+NUM_BASE_CONFIGS+NUM_XOREC_CPU_CONFIGS; ++it) {
    if (selected_xorec_ec_benchmarks.find("xorec-cpu") != selected_xorec_ec_benchmarks.end()) {
      auto bm_name = get_ec_benchmark_name("xorec-cpu", *it);
      auto bm_func = get_ec_benchmark_func("xorec-cpu");
      benchmarks.push_back({ bm_name, bm_func, *it });
    }
  }

  // XOR-EC Unified memory Pointer benchmarks
  for (; it != configs.begin()+NUM_BASE_CONFIGS+NUM_XOREC_CPU_CONFIGS+NUM_XOREC_UNIFIED_PTR_CONFIGS; ++it) {
    if (selected_xorec_ec_benchmarks.find("xorec-unified-ptr") != selected_xorec_ec_benchmarks.end()) {
      auto bm_name = get_ec_benchmark_name("xorec-unified-ptr", *it);
      auto bm_func = get_ec_benchmark_func("xorec-unified-ptr");
      benchmarks.push_back({ bm_name, bm_func, *it });
    }
  }

  // XOR-EC GPU Pointer benchmarks
  for (; it != configs.begin()+NUM_BASE_CONFIGS+NUM_XOREC_CPU_CONFIGS+NUM_XOREC_UNIFIED_PTR_CONFIGS+NUM_XOREC_GPU_PTR_CONFIGS; ++it) {
    if (selected_xorec_ec_benchmarks.find("xorec-gpu-ptr") != selected_xorec_ec_benchmarks.end()) {
      auto bm_name = get_ec_benchmark_name("xorec-gpu-ptr", *it);
      auto bm_func = get_ec_benchmark_func("xorec-gpu-ptr");
      benchmarks.push_back({ bm_name, bm_func, *it });
    }
  }

  for (; it != configs.end(); ++it) {
    if (selected_xorec_ec_benchmarks.find("xorec-gpu-cmp") != selected_xorec_ec_benchmarks.end()) {
      auto bm_name = get_ec_benchmark_name("xorec-gpu-cmp", *it);
      auto bm_func = get_ec_benchmark_func("xorec-gpu-cmp");
      benchmarks.push_back({ bm_name, bm_func, *it });
    }
  }
  return benchmarks;
}

/**
 * @brief "Zips" the performance benchmark names, functions, and configurations into a vector of tuples
 * 
 * @param configs 
 * @return std::vector<BenchmarkTuple> 
 */
static std::vector<BenchmarkTuple> get_perf_benchmarks(std::vector<BenchmarkConfig>& configs) {
  std::vector<BenchmarkTuple> benchmarks;
  for (auto config : configs) {
    for (auto& inp_name : selected_perf_benchmarks) {
      auto bm_name = get_perf_benchmark_name(inp_name);
      auto bm_func = get_perf_benchmark_func(inp_name);
      config.xorec_params.version = perf_benchmark_version.at(inp_name);
      benchmarks.push_back({ bm_name, bm_func, config });
    }
  }
  return benchmarks;
}

/**
 * @brief Initializes reporters and runs the EC benchmarks
 * 
 * @param configs 
 */
static void run_ec_benchmarks(std::vector<BenchmarkConfig>& configs) {
  if (configs.empty()) throw_error("No EC benchmark configurations found.");

  int argc = 2;
  char *argv[] = { (char*)"benchmark", (char*)"--benchmark_out=console" };

  std::vector<BenchmarkTuple> benchmarks = get_ec_benchmarks(configs);
  std::unique_ptr<BenchmarkProgressReporter> console_reporter = std::make_unique<BenchmarkProgressReporter>(NUM_ITERATIONS * benchmarks.size(), START_TIME);
  std::unique_ptr<BenchmarkCSVReporter> csv_reporter = std::make_unique<BenchmarkCSVReporter>(RAW_DIR + RESULT_DIR + EC_OUTPUT_FILE_NAME, OVERWRITE_FILE);

  benchmark::ClearRegisteredBenchmarks();

  for (auto [name, func, cfg] : benchmarks) {
    cfg.progress_reporter = console_reporter.get();
    
    benchmark::RegisterBenchmark(name, func, cfg)
      ->UseRealTime()
      ->Iterations(NUM_ITERATIONS);
  }

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return;
  benchmark::RunSpecifiedBenchmarks(console_reporter.get(), csv_reporter.get());
  benchmark::Shutdown();
}

/**
 * @brief Initializes reporters and runs the performance benchmarks
 * 
 * @param configs 
 */
static void run_perf_benchmarks(std::vector<BenchmarkConfig>& configs) {
  if (configs.empty()) return;

  int argc = 3;
  char *argv[] = { (char*)"benchmark", (char*)"--benchmark_out=console", (char*)"--benchmark_perf_counters=CYCLES,INSTRUCTIONS"};

  std::vector<BenchmarkTuple> benchmarks = get_perf_benchmarks(configs);
  // For the performance benchmarks we only report the runs, not the individual iterations
  std::unique_ptr<PerfBenchmarkProgressReporter> console_reporter = std::make_unique<PerfBenchmarkProgressReporter>(benchmarks.size(), START_TIME);
  std::unique_ptr<PerfBenchmarkCSVReporter> csv_reporter = std::make_unique<PerfBenchmarkCSVReporter>(RAW_DIR + RESULT_DIR + PERF_OUTPUT_FILE_NAME, OVERWRITE_FILE);

  benchmark::ClearRegisteredBenchmarks();

  for (auto [name, func, cfg] : benchmarks) {
    benchmark::RegisterBenchmark(name, func, cfg);
  }

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return;
  benchmark::RunSpecifiedBenchmarks(console_reporter.get(), csv_reporter.get());
  benchmark::Shutdown();
}

/**
 * @brief Ensures that the result directory exists, creating it if it does not
 */
static void ensure_result_dir() {
  namespace fs = std::filesystem;
  std::string dir = RAW_DIR + RESULT_DIR;

  if (!fs::exists(dir)) {
    fs::create_directories(dir);
  }
}

/**
 * @brief Populates the lost block indices, EC and performance configurations vectors
 * 
 * @param ec_configs 
 * @param lost_block_idxs 
 * @param perf_configs 
 */
void get_configs(std::vector<BenchmarkConfig>& ec_configs, std::vector<std::vector<uint32_t>>& lost_block_idxs, std::vector<BenchmarkConfig>& perf_configs) {
  if (RUN_EC_BM) get_ec_configs(ec_configs, lost_block_idxs);
  if (RUN_PERF_BM) get_perf_configs(perf_configs);
}

/**
 * @brief Runs all benchmarks based on the user-passed arguments
 * 
 * @param ec_configs 
 * @param perf_configs 
 */
void run_benchmarks(std::vector<BenchmarkConfig>& ec_configs, std::vector<BenchmarkConfig>& perf_configs) {
  START_TIME = std::chrono::system_clock::now();
  ensure_result_dir();
  if (RUN_EC_BM) run_ec_benchmarks(ec_configs);
  if (RUN_PERF_BM) run_perf_benchmarks(perf_configs);
}