/**
 * @file benchmark_suite.cpp
 * @brief Implementations of utility functions for parsing and validating command-line arguments
 */

#include "benchmark_cli.hpp"
#include "benchmark_suite.hpp"
#include "benchmark/benchmark.h"
#include "bm_config.hpp"
#include "console_reporter.hpp"
#include "csv_reporter.hpp"
#include "runners.hpp"
#include "utils.hpp"
#include "xorec_utils.hpp"
#include <filesystem>
#include <getopt.h>
#include <ranges>
#include <unordered_map>
#include <unordered_set>



constexpr const char* RAW_DIR = "../results/raw/";
std::string OUTPUT_FILE_NAME = "results.csv";

bool OVERWRITE_FILE = true;
int NUM_ITERATIONS = 10;

using BenchmarkTuple = std::tuple<std::string, BenchmarkFunction, BenchmarkConfig>;

std::unordered_set<std::string> selected_benchmarks;
std::unordered_set<XorecVersion> selected_xorec_versions;

const std::unordered_map<std::string, BenchmarkFunction> available_benchmarks = {
  { "cm256",              BM_CM256              },
  { "isal",               BM_ISAL               },
  { "leopard",            BM_Leopard            },
  { "xorec-cpu",          BM_XOREC              },
  { "xorec-unified-ptr",  BM_XOREC_UNIFIED_PTR  },
  { "xorec-gpu-ptr",      BM_XOREC_GPU_PTR      },
  { "xorec-gpu-cmp",      BM_XOREC_GPU_CMP      }
};

const std::unordered_map<std::string, std::string> benchmark_names = {
  { "cm256",              "CM256"                 },
  { "isal",               "ISA-L"                 },
  { "leopard",            "Leopard"               },
  { "xorec-cpu",          "XOR-EC (CPU)"          },
  { "xorec-unified-ptr",  "XOR-EC (Unified Ptr)"  },
  { "xorec-gpu-ptr",      "XOR-EC (GPU Ptr)"      },
  { "xorec-gpu-cmp",      "XOR-EC (GPU Cmp)"      }
};

const std::unordered_map<std::string, XorecVersion> available_xorec_versions = {
  { "scalar",   XorecVersion::Scalar   },
  { "avx",      XorecVersion::AVX      },
  { "avx2",     XorecVersion::AVX2     },
  { "avx512",   XorecVersion::AVX512   }
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
  if (available_benchmarks.find(name) != available_benchmarks.end()) {
    selected_benchmarks.insert(name);
  } else {
    throw_error("Error: Invalid benchmark name: " + name);
  }
}

static void inline add_xorec_version(std::string name) {
  if (available_xorec_versions.find(name) != available_xorec_versions.end()) {
    selected_xorec_versions.insert(available_xorec_versions.at(name));
  } else {
    throw_error("Error: Invalid xorec version name: " + name);
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
    { "file",           required_argument,  nullptr, 'f'  },
    { "append",         no_argument,        nullptr, 'a'  },
    { "iterations",     required_argument,  nullptr, 'i'  },
    { "base",           required_argument,  nullptr,  0   },
    { "xorec",          required_argument,  nullptr,  0   },
    { "simd",           required_argument,  nullptr,  0   },
    { nullptr,          0,                  nullptr,  0   }
  };

  int c;
  int option_index = 0;
  std::string flag;
  std::vector<std::string> args;

  while ((c = getopt_long(argc, argv, "hf:ai:", long_options, &option_index)) != -1) {
    switch (c) {
      case 'h':
        usage();
        break;
      case 'i':
        NUM_ITERATIONS = std::stoi(optarg);
        break;
      case 'f':
        OUTPUT_FILE_NAME = std::string(optarg);
        if (OUTPUT_FILE_NAME.find(".csv") == std::string::npos) {
          std::cerr << "Error: Output file must have .csv extension.\n";
          exit(0);
        }
        break;
      case 'a':
        OVERWRITE_FILE = false;
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
            add_xorec_version(arg);
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

  
  if (selected_benchmarks.empty()) {
    for (const auto& [name, _] : available_benchmarks) {
      add_benchmark(name);
    }
  }

  if (selected_xorec_versions.empty()) {
    for (const auto& [name, _] : available_xorec_versions) {
      add_xorec_version(name);
    }
  }
}


/**
 * @brief Initliazes the lost block indices used for benchmarking
 * 
 * @param lost_block_idxs A vector of vectors to store the lost block indices (should be empty)
 */
static void init_lost_block_idxs(std::vector<std::vector<uint32_t>>& lost_block_idxs) {
  // for ([[maybe_unused]] auto _ : VAR_BUFFER_SIZE) {
  //   std::vector<uint32_t> vec;
  //   select_lost_block_idxs(
  //     FIXED_NUM_RECOVERY_BLOCKS,
  //     FIXED_NUM_LOST_BLOCKS,
  //     FIXED_NUM_ORIGINAL_BLOCKS + FIXED_NUM_RECOVERY_BLOCKS,
  //     vec
  //   );
  //   lost_block_idxs.push_back(vec);
  // }

  // for (auto num_rec_blocks : VAR_NUM_RECOVERY_BLOCKS) {
  //   std::vector<uint32_t> vec;
  //   select_lost_block_idxs(
  //     num_rec_blocks,
  //     FIXED_NUM_LOST_BLOCKS,
  //     FIXED_NUM_ORIGINAL_BLOCKS + num_rec_blocks,
  //     vec
  //   );
  //   lost_block_idxs.push_back(vec);
  // }

  // for (auto num_lost_blocks : VAR_NUM_LOST_BLOCKS) {
  //   std::vector<uint32_t> vec;
  //   select_lost_block_idxs(
  //     FIXED_NUM_ORIGINAL_BLOCKS,
  //     num_lost_blocks,
  //     FIXED_NUM_ORIGINAL_BLOCKS + FIXED_NUM_ORIGINAL_BLOCKS,
  //     vec
  //   );
  //   lost_block_idxs.push_back(vec);
  // }
}

static void get_configs(std::vector<BenchmarkConfig>& configs) {
  if (configs.size() > 0) throw_error("Error: Configs vector should be empty.");
  
  for (auto buf_size : VAR_BUFFER_SIZE) {
    BenchmarkConfig config;
    config.data_size = buf_size;
    config.block_size = buf_size / FIXED_NUM_ORIGINAL_BLOCKS;
    config.num_lost_blocks = FIXED_NUM_LOST_BLOCKS;
    config.redundancy_ratio = FIXED_PARITY_RATIO;
    config.num_iterations = NUM_ITERATIONS;
    config.plot_id = 0;
    config.num_data_blocks = FIXED_NUM_ORIGINAL_BLOCKS;
    config.num_parity_blocks = FIXED_NUM_RECOVERY_BLOCKS;
    config.is_xorec_config = false;
    config.reporter = nullptr;
    configs.push_back(config);
  }

  for (auto num_rec_blocks : VAR_NUM_RECOVERY_BLOCKS) {
    BenchmarkConfig config;
    config.data_size = FIXED_BUFFER_SIZE;
    config.block_size = FIXED_BUFFER_SIZE / FIXED_NUM_ORIGINAL_BLOCKS;
    config.num_lost_blocks = FIXED_NUM_LOST_BLOCKS;
    config.redundancy_ratio = static_cast<double>(num_rec_blocks) / FIXED_NUM_ORIGINAL_BLOCKS;
    config.num_iterations = NUM_ITERATIONS;
    config.plot_id = 1;
    config.num_data_blocks = FIXED_NUM_ORIGINAL_BLOCKS;
    config.num_parity_blocks = num_rec_blocks;
    config.is_xorec_config = false;
    config.reporter = nullptr;
    configs.push_back(config);
  }

  for (auto num_lost_blocks : VAR_NUM_LOST_BLOCKS) {
    BenchmarkConfig config;
    config.data_size = FIXED_BUFFER_SIZE;
    config.block_size = FIXED_BUFFER_SIZE / FIXED_NUM_ORIGINAL_BLOCKS;
    config.num_lost_blocks = num_lost_blocks;
    config.redundancy_ratio = 1.0;
    config.num_iterations = NUM_ITERATIONS;
    config.plot_id = 2;
    config.num_data_blocks = FIXED_NUM_ORIGINAL_BLOCKS;
    config.num_parity_blocks = FIXED_NUM_ORIGINAL_BLOCKS;
    config.is_xorec_config = false;
    config.reporter = nullptr;
    configs.push_back(config);
  }
}

void get_benchmarks(std::vector<BenchmarkTuple>& benchmarks) {
  std::vector<BenchmarkConfig> configs;
  get_configs(configs);

  for (auto config : configs) {
    for (auto& inp_name : selected_benchmarks) {
      auto bm_name = benchmark_names.at(inp_name);
      auto bm_func = available_benchmarks.at(inp_name);
      if (inp_name == "xorec-cpu" || inp_name == "xorec-gpu-ptr" || inp_name == "xorec-unified-ptr") {
        config.is_xorec_config = true;
        if (inp_name == "xorec-unified-ptr") config.xorec_params.unified_mem = true;

        for (auto version : selected_xorec_versions) {
          config.xorec_params.version = version;
          bm_name += ", " + get_version_name(version);
          benchmarks.push_back({ bm_name, bm_func, config });
        }
      } else if (inp_name == "xorec-gpu-cmp") {
        config.is_xorec_config = true;
        benchmarks.push_back({ bm_name, bm_func, config });
      } else {
        benchmarks.push_back({ bm_name, bm_func, config });
      }
    }
  }
}

void run_benchmarks(int argc, char** argv) {
  auto start_time = std::chrono::system_clock::now();
  
  int new_argc = 2;
  char *new_argv[] = { (char*)"benchmark", (char*)"--benchmark_out=console" };
  
  parse_args(argc, argv);
  std::vector<BenchmarkTuple> benchmarks;
  get_benchmarks(benchmarks);

  std::unique_ptr<ConsoleReporter> console_reporter = std::make_unique<ConsoleReporter>(NUM_ITERATIONS * benchmarks.size(), start_time);
  std::unique_ptr<CSVReporter> csv_reporter = std::make_unique<CSVReporter>(RAW_DIR + OUTPUT_FILE_NAME, OVERWRITE_FILE);

  benchmark::ClearRegisteredBenchmarks();
  for (auto [name, func, cfg] : benchmarks) {
    cfg.reporter = console_reporter.get();

    benchmark::RegisterBenchmark(name, func, cfg)
      ->UseRealTime()
      ->Iterations(NUM_ITERATIONS);
  }

  benchmark::Initialize(&new_argc, new_argv);
  if (benchmark::ReportUnrecognizedArguments(new_argc, new_argv)) return;
  benchmark::RunSpecifiedBenchmarks(console_reporter.get(), csv_reporter.get());
  benchmark::Shutdown();
}