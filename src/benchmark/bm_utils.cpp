/**
 * @file benchmark_utils.cpp
 * @brief Implementations of utility functions for parsing and validating command-line arguments
 */

#include "bm_utils.hpp"
#include "benchmark/benchmark.h"
#include "bm_functions.hpp"
#include "utils.hpp"
#include <filesystem>
#include <getopt.h>

// Global variable definitions
constexpr const char* RAW_DIR = "../results/raw/";
std::string RESULT_DIR = "";
std::string EC_OUTPUT_FILE_NAME = "ec_results.csv";
std::string PERF_OUTPUT_FILE_NAME = "perf_results.csv";

bool OVERWRITE_FILE = true;
int NUM_ITERATIONS = 10;

int NUM_BASE_CONFIGS = 0;
int NUM_XOREC_CPU_CONFIGS = 0;
int NUM_XOREC_UNIFIED_PTR_CONFIGS = 0;
int NUM_XOREC_GPU_PTR_CONFIGS = 0;
int NUM_XOREC_GPU_CMP_CONFIGS = 0;

bool RUN_XOREC_SCALAR = false;
bool RUN_XOREC_AVX = false;
bool RUN_XOREC_AVX2 = false;
bool RUN_XOREC_AVX512 = false;

bool RUN_PERF_BM = false;
bool RUN_EC_BM = false;

bool INIT_BASE_CONFIGS = false;
bool INIT_XOREC_CPU_CONFIGS = false;
bool INIT_XOREC_UNIFIED_PTR_CONFIGS = false;
bool INIT_XOREC_GPU_PTR_CONFIGS = false;



enum class TouchUnifiedMemory {
  TOUCH_UNIFIED_MEM_TRUE = 0,
  TOUCH_UNIFIED_MEM_FALSE = 1,
  TOUCH_UNIFIED_MEM_ALL = 2
};

enum class XorecPrefetch {
  XOREC_PREFETCH = 0,
  XOREC_NO_PREFETCH = 1,
  XOREC_ALL_PREFETCH = 2
};

using BenchmarkTuple = std::tuple<std::string, BenchmarkFunction, BenchmarkConfig>;

TouchUnifiedMemory TOUCH_UNIFIED_MEM = TouchUnifiedMemory::TOUCH_UNIFIED_MEM_FALSE;
XorecPrefetch PREFETCH = XorecPrefetch::XOREC_NO_PREFETCH;


std::unordered_set<std::string> selected_base_ec_benchmarks;
std::unordered_set<std::string> selected_xorec_ec_benchmarks;

const std::unordered_map<std::string, BenchmarkFunction> available_base_ec_benchmarks = {
  { "cm256",                  BM_CM256                },
  { "isal",                   BM_ISAL                 },
  { "leopard",                BM_Leopard              },
  { "wirehair",               BM_Wirehair             }
};

const std::unordered_map<std::string, BenchmarkFunction> available_xorec_ec_benchmarks = {
  { "xorec",                  BM_XOREC              },
  { "xorec-unified-ptr",      BM_XOREC_UNIFIED_PTR  },
  { "xorec-gpu-ptr",          BM_XOREC_GPU_PTR      },
  { "xorec-gpu-cmp",          BM_XOREC_GPU_CMP      }
};

const std::unordered_map<std::string, std::string> ec_benchmark_names = {
  { "cm256",                "CM256"                 },
  { "isal",                 "ISA-L"                 },
  { "leopard",              "Leopard"               },
  { "wirehair",             "Wirehair"              },
  { "xorec",                "XOR-EC"                },
  { "xorec-unified-ptr",    "XOR-EC (Unified Ptr)"  },
  { "xorec-gpu-ptr",        "XOR-EC (GPU Ptr)"      },
  { "xorec-gpu-cmp",        "XOR-EC (GPU Cmp)"      }
};

std::unordered_set<std::string> selected_perf_benchmarks;

const std::unordered_map<std::string, BenchmarkFunction> available_perf_benchmarks = {
  { "perf-xorec-scalar", BM_XOR_BLOCKS_SCALAR },
  { "perf-xorec-avx",    BM_XOR_BLOCKS_AVX    },
  { "perf-xorec-avx2",   BM_XOR_BLOCKS_AVX2   },
  { "perf-xorec-avx512", BM_XOR_BLOCKS_AVX512 }
};

const std::unordered_map<std::string, std::string> perf_benchmark_names = {
  { "perf-xorec-scalar", "Theoretical XOR-EC (SCALAR)" },
  { "perf-xorec-avx",    "Theoretical XOR-EC (AVX)"    },
  { "perf-xorec-avx2",   "Theoretical XOR-EC (AVX2)"   },
  { "perf-xorec-avx512", "Theoretical XOR-EC (AVX512)" }
};


static void usage() {
  std::cerr << "Usage: ec-benchmark [options]\n\n"

            << " Help Option:\n"
            << "  -h, --help                              show this help message\n\n"
            
            << " Benchmark Options:\n"
            << "  -r, --result-dir=<dir_name>             specify output result subdirectory (inside /results/raw/),\n"
            << "                                          will be created if it doesn't exist\n"
            << "  -a, --append                            append results to the output file (default: overwrite)\n"
            << "  -b, --benchmark=<ec|perf|all>           specify the type of benchmark to run (default: all)\n\n"

            << " Erase Code Benchmarking Option\n"
            << "  -i, --iterations=<num>                  number of benchmark iterations (default 10)\n\n"
            << " Base Algorithm Selection:\n"
            << "      --cm256                             run the CM256 benchmark\n"
            << "      --isal                              run the ISA-L benchmark\n"
            << "      --leopard                           run the Leopard benchmark\n"
            << "      --wirehair                          run the Wirehair benchmark\n\n"

            << " XOR-EC Algorithm Selection:\n"
            << "      --xorec                             run the XOR-EC implementation (data buffer, parity buffer\n"
            << "                                          & computation on CPU)\n"
            << "      --xorec-unified-ptr                 run the XOR-EC implementation (data buffer in unified memory,\n"
            << "                                          parity buffer & computation on CPU)\n"
            << "      --xorec-gpu-ptr                     run the XOR-EC implementation (data buffer in GPU memory,\n"
            << "                                          parity buffer & computation on CPU)\n"
            << "      --xorec-gpu-cmp                     run the XOR-EC implementation (data buffer, parity buffer\n"
            << "                                          & computation in GPU memory, bitmap in CPU memory)\n\n"

            << " *If no algorithm is specified, all algorithms will be run.*\n\n"

            << " XOR-EC Version Options: (relevant if --xorec or --xorec-gpu-ptr specified)\n"
            << "      --scalar                            run the scalar XOR-EC implementation\n"
            << "      --avx                               run the AVX XOR-EC implementation\n"
            << "      --avx2                              run the AVX2 XOR-EC implementation\n"
            << "      --avx512                            run the AVX512 XOR-EC implementation\n\n"
            << " *If no versions are specified all 4 will be run.*\n\n"

            << " XOR-EC GPU Options: (relevant if --xorec-gpu-ptr or --xorec-gpu-cmp specified)\n"
            << "      --touch-unified-memory <true|false|all> whether to touch unified memory on the GPU before encoding/decoding\n"
            << "                                          (default: false)\n"
            << "      --prefetch <true|false|all>         whether to prefetch data blocks from unified memory to CPU memory, or fetch them on-demand,\n"
            << "                                          only relevant if --xorec-unified-ptr is specified (default: false)\n"

            << " Performance Benchmark Selection (if none are selected, none are run):\n"
            << "      --perf-xorec-scalar                 run the theoretical XOR-EC (SCALAR) performance benchmark\n"
            << "      --perf-xorec-avx                    run the theoretical XOR-EC (AVX) performance benchmark\n"
            << "      --perf-xorec-avx2                   run the theoretical XOR-EC (AVX2) performance benchmark\n"
            << "      --perf-xorec-avx512                 run the theoretical XOR-EC (AVX512) performance benchmark\n\n"
            << " *If no versions are specified all 4 will be run.*\n\n";
  exit(0);
}

static void inline add_benchmark(std::string name) {
  if (available_base_ec_benchmarks.find(name) != available_base_ec_benchmarks.end()) {
    selected_base_ec_benchmarks.insert(name);
  } else if (available_xorec_ec_benchmarks.find(name) != available_xorec_ec_benchmarks.end()) {
    selected_xorec_ec_benchmarks.insert(name);
  } else {
    selected_perf_benchmarks.insert(name);
  }
}

void parse_args(int argc, char** argv) {
  struct option long_options[] = {
    { "help",                 no_argument,        nullptr, 'h'  },
    { "result-dir",           required_argument,  nullptr, 'r'  },
    { "append",               no_argument,        nullptr, 'a'  },
    { "benchmark",            required_argument,  nullptr, 'b'  },

    { "cm256",                no_argument,        nullptr,  0   },
    { "isal",                 no_argument,        nullptr,  0   },
    { "leopard",              no_argument,        nullptr,  0   },
    { "wirehair",             no_argument,        nullptr,  0   },
    
    { "iterations",           required_argument,  nullptr, 'i'  },

    { "xorec",                no_argument,        nullptr,  0   },
    { "xorec-unified-ptr",    no_argument,        nullptr,  0   },
    { "xorec-gpu-ptr",        no_argument,        nullptr,  0   },
    { "xorec-gpu-cmp",        no_argument,        nullptr,  0   },

    { "scalar",               no_argument,        nullptr,  0   },
    { "avx",                  no_argument,        nullptr,  0   },
    { "avx2",                 no_argument,        nullptr,  0   },
    { "avx512",               no_argument,        nullptr,  0   },

    { "touch-unified-memory",     required_argument,  nullptr,  0   },
    { "prefetch",             required_argument,  nullptr,  0   },

    { "perf-xorec-scalar",    no_argument,        nullptr,  0   },
    { "perf-xorec-avx",       no_argument,        nullptr,  0   },
    { "perf-xorec-avx2",      no_argument,        nullptr,  0   },
    { "perf-xorec-avx512",    no_argument,        nullptr,  0   },
    { nullptr,                0,                  nullptr,  0   }
  };


  int c;
  int option_index = 0;
  std::string flag;
  std::string arg;

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
        arg = to_lower(std::string(optarg));
        if (arg == "ec") {
          RUN_EC_BM = true;
          RUN_PERF_BM = false;
        } else if (arg == "perf") {
          RUN_EC_BM = false;
          RUN_PERF_BM = true;
        } else if (arg == "all") {
          RUN_PERF_BM = true;
          RUN_EC_BM = true;
        } else {
          std::cerr << "Error: --benchmark option must be either 'ec', 'perf', or 'all'.\n";
          exit(0);
        }
        break;
      case 0:
        flag = std::string(long_options[option_index].name);
        
        if (flag == "scalar") {
          RUN_XOREC_SCALAR = true;
        } else if (flag == "avx") {
          RUN_XOREC_AVX = true;
        } else if (flag == "avx2") {
          RUN_XOREC_AVX2 = true;
        } else if (flag == "avx512") {
          RUN_XOREC_AVX512 = true;
        } else if (flag == "touch-unified-memory") {
          arg = to_lower(std::string(optarg));
          if (arg == "true" || arg == "1") {
            TOUCH_UNIFIED_MEM = TouchUnifiedMemory::TOUCH_UNIFIED_MEM_TRUE;
          } else if (arg == "false" || arg == "0") {
            TOUCH_UNIFIED_MEM = TouchUnifiedMemory::TOUCH_UNIFIED_MEM_FALSE;
          } else if (arg == "all") {
            TOUCH_UNIFIED_MEM = TouchUnifiedMemory::TOUCH_UNIFIED_MEM_ALL;
          } else {
            std::cerr << "Error: --touch-gpu-memory option must be either 'true', 'false', or 'all'.\n";
            exit(0);
          }
        }  else if (flag == "prefetch"){
          arg = to_lower(std::string(optarg));
          if (arg == "true" || arg == "1") {
            PREFETCH = XorecPrefetch::XOREC_PREFETCH;
          } else if (arg == "false" || arg == "0") {
            PREFETCH = XorecPrefetch::XOREC_NO_PREFETCH;
          } else if (arg == "all") {
            PREFETCH = XorecPrefetch::XOREC_ALL_PREFETCH;
          } else {
            std::cerr << "Error: --prefetch option must be either 'true', 'false', or 'all'.\n";
            exit(0);
          }
        } else {
          add_benchmark(flag);
        }
        break;
      default:
        usage();
        exit(0);
    }
  }

  if (!RUN_XOREC_SCALAR && !RUN_XOREC_AVX && !RUN_XOREC_AVX2 && !RUN_XOREC_AVX512) {
    RUN_XOREC_SCALAR = true;
    RUN_XOREC_AVX = true;
    RUN_XOREC_AVX2 = true;
    RUN_XOREC_AVX512 = true;
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
}

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

static void get_xorec_unified_ptr_configs(std::vector<BenchmarkConfig>& configs) {
  if (!INIT_XOREC_CPU_CONFIGS) throw_error("XOR-EC CPU configurations must be initialized before XOR-EC GPU Pointer configurations.");

  std::vector<BenchmarkConfig> prefetch_configs;

  for (int i = NUM_BASE_CONFIGS; i < NUM_BASE_CONFIGS+NUM_XOREC_CPU_CONFIGS; ++i) {
    BenchmarkConfig config = configs[i];
    config.xorec_params.unified_mem = true;

    if (PREFETCH == XorecPrefetch::XOREC_PREFETCH  || PREFETCH == XorecPrefetch::XOREC_ALL_PREFETCH) {
      config.xorec_params.prefetch = true;
      prefetch_configs.push_back(config);
    }

    if (PREFETCH == XorecPrefetch::XOREC_NO_PREFETCH || PREFETCH == XorecPrefetch::XOREC_ALL_PREFETCH) {
      config.xorec_params.prefetch = false;
      prefetch_configs.push_back(config);
    }
  }

  for (BenchmarkConfig config : prefetch_configs) {
    if (TOUCH_UNIFIED_MEM == TouchUnifiedMemory::TOUCH_UNIFIED_MEM_TRUE || TOUCH_UNIFIED_MEM == TouchUnifiedMemory::TOUCH_UNIFIED_MEM_ALL) {
      config.xorec_params.touch_unified_mem = true;
      configs.push_back(config);
      ++NUM_XOREC_UNIFIED_PTR_CONFIGS;
    }

    if (TOUCH_UNIFIED_MEM == TouchUnifiedMemory::TOUCH_UNIFIED_MEM_FALSE || TOUCH_UNIFIED_MEM == TouchUnifiedMemory::TOUCH_UNIFIED_MEM_ALL) {
      config.xorec_params.touch_unified_mem = false;
      configs.push_back(config);
      ++NUM_XOREC_UNIFIED_PTR_CONFIGS;
    }
  }
  
  INIT_XOREC_UNIFIED_PTR_CONFIGS = true;
}

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
      false,
      { XorecVersion::Scalar, false, false, false, false, false },
      nullptr
    };
    configs.push_back(config);
  }
}

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

static BenchmarkFunction get_ec_benchmark_func(const std::string& name) {
  if (available_base_ec_benchmarks.find(name) != available_base_ec_benchmarks.end()) {
    return available_base_ec_benchmarks.at(name);
  } else {
    return available_xorec_ec_benchmarks.at(name);
  }
}


static std::string get_perf_benchmark_name(const std::string inp_name) {
  return perf_benchmark_names.at(inp_name);
}

static BenchmarkFunction get_perf_benchmark_func(const std::string& name) {
  return available_perf_benchmarks.at(name);
}


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
    if (selected_xorec_ec_benchmarks.find("xorec") != selected_xorec_ec_benchmarks.end()) {
      auto bm_name = get_ec_benchmark_name("xorec", *it);
      auto bm_func = get_ec_benchmark_func("xorec");
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

static std::vector<BenchmarkTuple> get_perf_benchmarks(std::vector<BenchmarkConfig>& configs) {
  std::vector<BenchmarkTuple> benchmarks;
  for (auto config : configs) {
    for (auto& inp_name : selected_perf_benchmarks) {
      auto bm_name = get_perf_benchmark_name(inp_name);
      auto bm_func = get_perf_benchmark_func(inp_name);
      benchmarks.push_back({ bm_name, bm_func, config });
    }
  }
  return benchmarks;
}


static void run_ec_benchmarks(std::vector<BenchmarkConfig>& configs) {
  if (configs.empty()) throw_error("No EC benchmark configurations found.");

  int argc = 2;
  char *argv[] = { (char*)"benchmark", (char*)"--benchmark_out=console" };

  std::vector<BenchmarkTuple> benchmarks = get_ec_benchmarks(configs);
  std::unique_ptr<BenchmarkProgressReporter> console_reporter = std::make_unique<BenchmarkProgressReporter>(NUM_ITERATIONS * benchmarks.size());
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


static void run_perf_benchmarks(std::vector<BenchmarkConfig>& configs) {
  if (configs.empty()) return;

  int argc = 3;
  char *argv[] = { (char*)"benchmark", (char*)"--benchmark_out=console", (char*)"--benchmark_perf_counters=CYCLES,INSTRUCTIONS"};

  std::vector<BenchmarkTuple> benchmarks = get_perf_benchmarks(configs);
  // For the performance benchmarks we only report the runs, not the individual iterations
  std::unique_ptr<PerfBenchmarkProgressReporter> console_reporter = std::make_unique<PerfBenchmarkProgressReporter>(benchmarks.size());
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

static void ensure_result_dir() {
  namespace fs = std::filesystem;
  std::string dir = RAW_DIR + RESULT_DIR;

  if (!fs::exists(dir)) {
    fs::create_directories(dir);
  }
}



void get_configs(std::vector<BenchmarkConfig>& ec_configs, std::vector<std::vector<uint32_t>>& lost_block_idxs, std::vector<BenchmarkConfig>& perf_configs) {
  if (RUN_EC_BM) get_ec_configs(ec_configs, lost_block_idxs);
  if (RUN_PERF_BM) get_perf_configs(perf_configs);
}

void run_benchmarks(std::vector<BenchmarkConfig>& ec_configs, std::vector<BenchmarkConfig>& perf_configs) {
  ensure_result_dir();
  if (RUN_EC_BM) run_ec_benchmarks(ec_configs);
  if (RUN_PERF_BM) run_perf_benchmarks(perf_configs);
}