/**
 * @file benchmark_utils.cpp
 * @brief Implementations of utility functions for parsing and validating command-line arguments
 */

#include "benchmark/benchmark.h"
#include "benchmark_utils.hpp"
#include "benchmark_functions.hpp"
#include "utils.hpp"
#include <cmath>
#include <iostream>
#include <getopt.h>

// Global variable definitions
constexpr const char* OUTPUT_FILE_DIR = "../results/raw/";
std::string output_file_name = "benchmark_results.csv";
bool OVERWRITE_FILE = true;
int NUM_ITERATIONS = 10;
uint32_t NUM_BASE_CONFIGS = 0;


enum class TouchGPUMemory {
  TOUCH_GPU_MEM_TRUE = 0,
  TOUCH_GPU_MEM_FALSE = 1,
  TOUCH_GPU_MEM_ALL = 2
};

TouchGPUMemory TOUCH_GPU_MEM = TouchGPUMemory::TOUCH_GPU_MEM_FALSE;


std::unordered_set<std::string> selected_cpu_benchmarks;
std::unordered_set<std::string> selected_gpu_benchmarks;

const std::unordered_map<std::string, BenchmarkFunction> available_cpu_benchmarks = {
  { "cm256",                  BM_CM256                },
  { "isal",                   BM_ISAL                 },
  { "leopard",                BM_Leopard              },
  { "wirehair",               BM_Wirehair             },
  { "xorec-scalar",           BM_XOREC_SCALAR         },
  { "xorec-avx",              BM_XOREC_AVX            },
  { "xorec-avx2",             BM_XOREC_AVX2           },
};

const std::unordered_map<std::string, BenchmarkFunction> available_gpu_benchmarks = {
  { "xorec-scalar-gpu-ptr",   BM_XOREC_SCALAR_GPU_PTR },
  { "xorec-avx-gpu-ptr",      BM_XOREC_AVX_GPU_PTR    },
  { "xorec-avx2-gpu-ptr",     BM_XOREC_AVX2_GPU_PTR   },
  { "xorec-gpu-cmp",          BM_XOREC_GPU_CMP        }
};

const std::unordered_map<std::string, std::string> benchmark_names = {
  { "cm256",                "CM256"                         },
  { "isal",                 "ISA-L"                         },
  { "leopard",              "Leopard"                       },
  { "wirehair",             "Wirehair"                      },
  { "xorec-scalar",         "XOR-EC (Scalar)"               },
  { "xorec-avx",            "XOR-EC (AVX)"                  },
  { "xorec-avx2",           "XOR-EC (AVX2)"                 },
  { "xorec-scalar-gpu-ptr", "XOR-EC (Scalar, GPU Pointer)"  },
  { "xorec-avx-gpu-ptr",    "XOR-EC (AVX, GPU Pointer)"     },
  { "xorec-avx2-gpu-ptr",   "XOR-EC (AVX2, GPU Pointer)"    },
  { "xor-ec-gpu-cmp",       "XOR-EC (GPU Computation)"      }
};

bool is_gpu_benchmark(const std::string& name) {
  return available_gpu_benchmarks.find(name) != available_gpu_benchmarks.end();
}

static void usage() {
  std::cerr << "Usage: ec-benchmark [options]\n\n"

            << " Help Option:\n"
            << "  -h, --help                              show this help message\n\n"
            
            << " Benchmark Options:\n"
            << "  -i, --iterations=<num>                  number of benchmark iterations (default 10)\n"
            << "  -f, --file=<file_name>                  specify output file name\n"
            << "  -a, --append                            append results to the output file (default: overwrite)\n"
            << "      --touch-gpu-memory <true|false|all> whether to touch GPU memory before encoding/decoding\n"
            << "                                          (default: false)\n"
            << "                                          *only relevant if a GPU benchmark was specified*\n\n"
      
            << " CPU Algorithm Selection:\n"
            << "      --cm256                             run the CM256 benchmark\n"
            << "      --isal                              run the ISA-L benchmark\n"
            << "      --leopard                           run the Leopard benchmark\n"
            << "      --wirehair                          run the Wirehair benchmark\n"

            << "      --xorec-scalar                      run the scalar XOR-EC implementation (data buffer on CPU)\n"
            << "      --xorec-avx                         run the AVX XOR-EC implementation (data buffer on CPU)\n"
            << "      --xorec-avx2                        run the AVX2 XOR-EC implementation (data buffer on CPU)\n\n"

            << " GPU Algorithm Selection:\n"
            << "      --xorec-scalar-gpu-ptr              run the scalar XOR-EC implementation (data buffer on GPU)\n"
            << "      --xorec-avx-gpu-ptr                 run the AVX XOR-EC implementation (data buffer on GPU)\n"
            << "      --xorec-avx2-gpu-ptr                run the AVX2 XOR-EC implementation (data buffer on GPU)\n"

            << "      --xorec-scalar-gpu-cmp              run the scalar XOR-EC implementation (data buffer,\n"
            << "                                          parity buffer & computation on GPU)\n"
            << "      --xorec-avx-gpu-cmp                 run the AVX XOR-EC implementation (data buffer,\n"
            << "                                          parity buffer & computation on GPU)\n"
            << "      --xorec-avx2-gpu-cmp                run the AVX2 XOR-EC implementation (data buffer,\n"
            << "                                          parity buffer & computation on GPU)\n"
            
            << " *If no algorithm is specified, all algorithms will be run.*\n";
  exit(0);
}

static void inline add_benchmark(std::string name) {
  if (available_gpu_benchmarks.find(name) != available_gpu_benchmarks.end()) {
    selected_gpu_benchmarks.insert(name);
  } else if (available_cpu_benchmarks.find(name) != available_cpu_benchmarks.end()) {
    selected_cpu_benchmarks.insert(name);
  } else {
    throw_error("Invalid benchmark name: " + name);
  }
}

void parse_args(int argc, char** argv) {
  struct option long_options[] = {
    { "help",                 no_argument,        nullptr, 'h'  },
    { "iterations",           required_argument,  nullptr, 'i'  },
    { "file",                 required_argument,  nullptr, 'f'  },
    { "append",               no_argument,        nullptr, 'a'  },
    { "touch-gpu-memory",     required_argument,  nullptr,  0   },
    { "cm256",                no_argument,        nullptr,  0   },
    { "isal",                 no_argument,        nullptr,  0   },
    { "leopard",              no_argument,        nullptr,  0   },
    { "wirehair",             no_argument,        nullptr,  0   },
    { "xorec",                no_argument,        nullptr,  0   },
    { "xorec-scalar",         no_argument,        nullptr,  0   },
    { "xorec-avx",            no_argument,        nullptr,  0   },
    { "xorec-avx2",           no_argument,        nullptr,  0   },
    { "xorec-scalar-gpu-ptr", no_argument,        nullptr,  0   },
    { "xorec-avx-gpu-ptr",    no_argument,        nullptr,  0   },
    { "xorec-avx2-gpu-ptr",   no_argument,        nullptr,  0   },
    { "xorec-gpu-cmp",        no_argument,        nullptr,  0   },
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

        if (flag == "touch-gpu-memory") {
          auto arg = to_lower(std::string(optarg));
          if (arg == "true" || arg == "1") {
            TOUCH_GPU_MEM = TouchGPUMemory::TOUCH_GPU_MEM_TRUE;
          } else if (arg == "false" || arg == "0") {
            TOUCH_GPU_MEM = TouchGPUMemory::TOUCH_GPU_MEM_FALSE;
          } else if (arg == "all") {
            TOUCH_GPU_MEM = TouchGPUMemory::TOUCH_GPU_MEM_ALL;
          } else {
            std::cerr << "Error: --touch-gpu-memory option must be either 'true', 'false', or 'all'.\n";
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

  if (selected_cpu_benchmarks.empty() && selected_gpu_benchmarks.empty()) {
    for (const auto& [name, _] : available_cpu_benchmarks) {
      add_benchmark(name);
    }

    for (const auto& [name, _] : available_gpu_benchmarks) {
      add_benchmark(name);
    }
  }
}

static void init_lost_block_idxs(std::vector<std::vector<uint32_t>>& lost_block_idxs) {
  for (auto _ : VAR_BUFFER_SIZE) {
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

static void get_gpu_configs(std::vector<BenchmarkConfig>& configs, uint32_t num_base_configs) {
  for (uint32_t i = 0; i < num_base_configs; ++i) {
    BenchmarkConfig base_config = configs[i];
    base_config.gpu_mem = true;

    if (TOUCH_GPU_MEM == TouchGPUMemory::TOUCH_GPU_MEM_TRUE || TOUCH_GPU_MEM == TouchGPUMemory::TOUCH_GPU_MEM_ALL) {
      BenchmarkConfig config = base_config;
      config.touch_gpu_mem = true;
      configs.push_back(config);
    }

    if (TOUCH_GPU_MEM == TouchGPUMemory::TOUCH_GPU_MEM_FALSE || TOUCH_GPU_MEM == TouchGPUMemory::TOUCH_GPU_MEM_ALL) {
      BenchmarkConfig config = base_config;
      config.touch_gpu_mem = false;
      configs.push_back(config);
    }
  }
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
      { FIXED_NUM_ORIGINAL_BLOCKS, FIXED_NUM_RECOVERY_BLOCKS },
      false,
      false,
      nullptr
    };
    configs.push_back(config);
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
      { FIXED_NUM_ORIGINAL_BLOCKS, num_rec_blocks },
      false,
      false,
      nullptr
    };
    configs.push_back(config);
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
      { FIXED_NUM_ORIGINAL_BLOCKS, FIXED_NUM_ORIGINAL_BLOCKS },
      false,
      false,
      nullptr
    };
    configs.push_back(config);
  }
}

void get_configs(std::vector<BenchmarkConfig>& configs, std::vector<std::vector<uint32_t>>& lost_block_idxs) {
  init_lost_block_idxs(lost_block_idxs);

  get_base_configs(configs, lost_block_idxs);

  NUM_BASE_CONFIGS = configs.size();
  
  if (selected_gpu_benchmarks.size() > 0) {
    get_gpu_configs(configs, NUM_BASE_CONFIGS);
  }
}

std::string get_benchmark_name(const std::string inp_name, bool gpu_mem, bool touch_gpu_mem) {
  std::string name = benchmark_names.at(inp_name);
  if (gpu_mem && touch_gpu_mem) {
    name += " (memory touched)";
  } else if (gpu_mem && !touch_gpu_mem) {
    name += " (memory untouched)";
  }
  return name;
}

BenchmarkFunction get_benchmark_func(const std::string& name) {
  if (is_gpu_benchmark(name)) {
    return available_gpu_benchmarks.at(name);
  } else {
    return available_cpu_benchmarks.at(name);
  }
}


static void register_benchmarks(std::vector<BenchmarkConfig>& configs, BenchmarkProgressReporter *console_reporter) {
  for (auto& config: configs) {
    config.progress_reporter = console_reporter;
    if (!config.gpu_mem) {
      for (auto& inp_name : selected_cpu_benchmarks) {
        auto bm_name = get_benchmark_name(inp_name, config.gpu_mem, config.touch_gpu_mem);
        auto bm_func = get_benchmark_func(inp_name);
        benchmark::RegisterBenchmark(bm_name, bm_func, config)->UseManualTime()->Iterations(config.num_iterations);
      }
    } else {
      for (auto& inp_name : selected_gpu_benchmarks) {
        auto bm_name = get_benchmark_name(inp_name, config.gpu_mem, config.touch_gpu_mem);
        auto bm_func = get_benchmark_func(inp_name);
        benchmark::RegisterBenchmark(bm_name, bm_func, config)->UseManualTime()->Iterations(config.num_iterations);
      }
    }
  }
}

void run_benchmarks(std::vector<BenchmarkConfig>& configs) {
  if (configs.empty()) throw_error("No benchmark configurations found.");

  int argc = 2;
  char *argv[] = { (char*)"benchmark", (char*)"--benchmark_out=console" };

  int tot_iterations = NUM_ITERATIONS * (NUM_BASE_CONFIGS * selected_cpu_benchmarks.size()) + ((configs.size() - NUM_BASE_CONFIGS) * selected_gpu_benchmarks.size()); 

  std::unique_ptr<BenchmarkProgressReporter> console_reporter = std::make_unique<BenchmarkProgressReporter>(tot_iterations);
  std::unique_ptr<BenchmarkCSVReporter> csv_reporter = std::make_unique<BenchmarkCSVReporter>(OUTPUT_FILE_DIR + output_file_name, OVERWRITE_FILE);


  register_benchmarks(configs, console_reporter.get());

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return;
  benchmark::RunSpecifiedBenchmarks(console_reporter.get(), csv_reporter.get());
  benchmark::Shutdown();
}