/**
 * @file benchmark_utils.cpp
 * @brief Implementations of utility functions for parsing and validating command-line arguments
 */

#include "benchmark_utils.hpp"
#include "benchmark/benchmark.h"
#include "benchmark_functions.hpp"
#include "utils.hpp"
#include <getopt.h>

// Global variable definitions
constexpr const char* OUTPUT_FILE_DIR = "../results/raw/";
std::string output_file_name = "benchmark_results.csv";

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


std::unordered_set<std::string> selected_base_benchmarks;
std::unordered_set<std::string> selected_xorec_benchmarks;

const std::unordered_map<std::string, BenchmarkFunction> available_base_benchmarks = {
  { "cm256",                  BM_CM256                },
  { "isal",                   BM_ISAL                 },
  { "leopard",                BM_Leopard              },
  { "wirehair",               BM_Wirehair             }
};

const std::unordered_map<std::string, BenchmarkFunction> available_xorec_benchmarks = {
  { "xorec",                  BM_XOREC              },
  { "xorec-unified-ptr",      BM_XOREC_UNIFIED_PTR  },
  { "xorec-gpu-ptr",          BM_XOREC_GPU_PTR      },
  { "xorec-gpu-cmp",          BM_XOREC_GPU_CMP      }
};

const std::unordered_map<std::string, std::string> benchmark_names = {
  { "cm256",                "CM256"                 },
  { "isal",                 "ISA-L"                 },
  { "leopard",              "Leopard"               },
  { "wirehair",             "Wirehair"              },
  { "xorec",                "XOR-EC"                },
  { "xorec-unified-ptr",    "XOR-EC (Unified Ptr)"  },
  { "xorec-gpu-ptr",        "XOR-EC (GPU Ptr)"      },
  { "xorec-gpu-cmp",        "XOR-EC (GPU Cmp)"      }
};


static void usage() {
  std::cerr << "Usage: ec-benchmark [options]\n\n"

            << " Help Option:\n"
            << "  -h, --help                              show this help message\n\n"
            
            << " Benchmark Options:\n"
            << "  -i, --iterations=<num>                  number of benchmark iterations (default 10)\n"
            << "  -f, --file=<file_name>                  specify output file name\n"
            << "  -a, --append                            append results to the output file (default: overwrite)\n\n"
      
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
            << "      --avx2                              run the AVX2 XOR-EC implementation\n\n"
            << " *If no versions are specified all 3 will be run.*\n\n"

            << " XOR-EC GPU Options: (relevant if --xorec-gpu-ptr or --xorec-gpu-cmp specified)\n"
            << "      --touch-unified-memory <true|false|all> whether to touch unified memory on the GPU before encoding/decoding\n"
            << "                                          (default: false)\n"
            << "      --prefetch <true|false|all>         whether to prefetch data blocks from unified memory to CPU memory, or fetch them on-demand,\n"
            << "                                          only relevant if --xorec-unified-ptr is specified (default: false)\n\n";
  exit(0);
}

static void inline add_benchmark(std::string name) {
  if (available_base_benchmarks.find(name) != available_base_benchmarks.end()) {
    selected_base_benchmarks.insert(name);
  } else {
    selected_xorec_benchmarks.insert(name);
  }
}

void parse_args(int argc, char** argv) {
  struct option long_options[] = {
    { "help",                 no_argument,        nullptr, 'h'  },
    { "iterations",           required_argument,  nullptr, 'i'  },
    { "file",                 required_argument,  nullptr, 'f'  },
    { "append",               no_argument,        nullptr, 'a'  },
    { "cm256",                no_argument,        nullptr,  0   },
    { "isal",                 no_argument,        nullptr,  0   },
    { "leopard",              no_argument,        nullptr,  0   },
    { "wirehair",             no_argument,        nullptr,  0   },

    { "xorec",                no_argument,        nullptr,  0   },
    { "xorec-unified-ptr",    no_argument,        nullptr,  0   },
    { "xorec-gpu-ptr",        no_argument,        nullptr,  0   },
    { "xorec-gpu-cmp",        no_argument,        nullptr,  0   },

    { "scalar",               no_argument,        nullptr,  0   },
    { "avx",                  no_argument,        nullptr,  0   },
    { "avx2",                 no_argument,        nullptr,  0   },

    { "touch-unified-memory",     required_argument,  nullptr,  0   },
    { "prefetch",             required_argument,  nullptr,  0   },
    { nullptr,                0,                  nullptr,  0   }
  };


  int c;
  int option_index = 0;
  std::string flag;

  while ((c = getopt_long(argc, argv, "hs:b:l:r:i:", long_options, &option_index)) != -1) {
    switch (c) {
      case 'h':
        usage();
        break;
      case 'i':
        NUM_ITERATIONS = std::stoi(optarg);
        break;
      case 'f':
        output_file_name = std::string(optarg);
        break;
      case 'a':
        OVERWRITE_FILE = false;
        break;
      case 0:
        flag = std::string(long_options[option_index].name);
        
        if (flag == "scalar") {
          RUN_XOREC_SCALAR = true;
        } else if (flag == "avx") {
          RUN_XOREC_AVX = true;
        } else if (flag == "avx2") {
          RUN_XOREC_AVX2 = true;
        } else if (flag == "touch-unified-memory") {
          auto arg = to_lower(std::string(optarg));
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
          auto arg = to_lower(std::string(optarg));
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

  if (!RUN_XOREC_SCALAR && !RUN_XOREC_AVX && !RUN_XOREC_AVX2) {
    RUN_XOREC_SCALAR = true;
    RUN_XOREC_AVX = true;
    RUN_XOREC_AVX2 = true;
  }

  if (selected_base_benchmarks.empty() && selected_xorec_benchmarks.empty()) {
    for (const auto& [name, _] : available_base_benchmarks) {
      add_benchmark(name);
    }

    for (const auto& [name, _] : available_xorec_benchmarks) {
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

void get_configs(std::vector<BenchmarkConfig>& configs, std::vector<std::vector<uint32_t>>& lost_block_idxs) {
  init_lost_block_idxs(lost_block_idxs);

  get_base_configs(configs, lost_block_idxs);

  if (selected_xorec_benchmarks.size() > 0) {
    get_xorec_cpu_configs(configs);
    get_xorec_unified_ptr_configs(configs);
    get_xorec_gpu_ptr_configs(configs);
    get_xorec_gpu_cmp_configs(configs);
  }
}

std::string get_benchmark_name(const std::string inp_name, BenchmarkConfig config) {
  std::string name = benchmark_names.at(inp_name);
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
      name += ", prefetched";
    }
  }

  return name;
}

BenchmarkFunction get_benchmark_func(const std::string& name) {
  if (available_base_benchmarks.find(name) != available_base_benchmarks.end()) {
    return available_base_benchmarks.at(name);
  } else {
    return available_xorec_benchmarks.at(name);
  }
}


static std::vector<BenchmarkTuple> get_benchmarks(std::vector<BenchmarkConfig>& configs) {
  std::vector<BenchmarkTuple> benchmarks;

  auto it = configs.begin();
  // Base benchmarks
  for (; it != configs.begin()+NUM_BASE_CONFIGS; ++it) {
    for (auto& inp_name : selected_base_benchmarks) {
      auto bm_name = get_benchmark_name(inp_name, *it);
      auto bm_func = get_benchmark_func(inp_name);
      benchmarks.push_back({ bm_name, bm_func, *it });
    }
  }

  // XOR-EC CPU benchmarks
  for (; it != configs.begin()+NUM_BASE_CONFIGS+NUM_XOREC_CPU_CONFIGS; ++it) {
    if (selected_xorec_benchmarks.find("xorec") != selected_xorec_benchmarks.end()) {
      auto bm_name = get_benchmark_name("xorec", *it);
      auto bm_func = get_benchmark_func("xorec");
      benchmarks.push_back({ bm_name, bm_func, *it });
    }
  }

  // XOR-EC Unified memory Pointer benchmarks
  for (; it != configs.begin()+NUM_BASE_CONFIGS+NUM_XOREC_CPU_CONFIGS+NUM_XOREC_UNIFIED_PTR_CONFIGS; ++it) {
    if (selected_xorec_benchmarks.find("xorec-unified-ptr") != selected_xorec_benchmarks.end()) {
      auto bm_name = get_benchmark_name("xorec-unified-ptr", *it);
      auto bm_func = get_benchmark_func("xorec-unified-ptr");
      benchmarks.push_back({ bm_name, bm_func, *it });
    }
  }

  // XOR-EC GPU Pointer benchmarks
  for (; it != configs.begin()+NUM_BASE_CONFIGS+NUM_XOREC_CPU_CONFIGS+NUM_XOREC_UNIFIED_PTR_CONFIGS+NUM_XOREC_GPU_PTR_CONFIGS; ++it) {
    if (selected_xorec_benchmarks.find("xorec-gpu-ptr") != selected_xorec_benchmarks.end()) {
      auto bm_name = get_benchmark_name("xorec-gpu-ptr", *it);
      auto bm_func = get_benchmark_func("xorec-gpu-ptr");
      benchmarks.push_back({ bm_name, bm_func, *it });
    }
  }

  for (; it != configs.end(); ++it) {
    if (selected_xorec_benchmarks.find("xorec-gpu-cmp") != selected_xorec_benchmarks.end()) {
      auto bm_name = get_benchmark_name("xorec-gpu-cmp", *it);
      auto bm_func = get_benchmark_func("xorec-gpu-cmp");
      benchmarks.push_back({ bm_name, bm_func, *it });
    }
  }
  return benchmarks;
}

void run_benchmarks(std::vector<BenchmarkConfig>& configs) {
  if (configs.empty()) throw_error("No benchmark configurations found.");

  int argc = 2;
  char *argv[] = { (char*)"benchmark", (char*)"--benchmark_out=console" };

  std::vector<BenchmarkTuple> benchmarks = get_benchmarks(configs);
  std::unique_ptr<BenchmarkProgressReporter> console_reporter = std::make_unique<BenchmarkProgressReporter>(NUM_ITERATIONS * benchmarks.size());
  std::unique_ptr<BenchmarkCSVReporter> csv_reporter = std::make_unique<BenchmarkCSVReporter>(OUTPUT_FILE_DIR + output_file_name, OVERWRITE_FILE);

  for (auto [name, func, conf] : benchmarks) {
    conf.progress_reporter = console_reporter.get();
    
    benchmark::RegisterBenchmark(name, func, conf)
      ->UseRealTime()
      ->Iterations(NUM_ITERATIONS);
  }

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return;
  benchmark::RunSpecifiedBenchmarks(console_reporter.get(), csv_reporter.get());
  benchmark::Shutdown();
}