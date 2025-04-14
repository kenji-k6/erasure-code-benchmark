/**
 * @file benchmark_suite.cpp
 * @brief Implementations of utility functions for parsing and validating command-line arguments
 */

#include "benchmark_cli.hpp"
#include "benchmark_suite.hpp"
#include "benchmark/benchmark.h"
#include "console_reporter.hpp"
#include "csv_reporter.hpp"
#include "runners.hpp"
#include "utils.hpp"
#include <filesystem>
#include <getopt.h>
#include <ranges>
#include <unordered_set>


constexpr const char* RAW_DIR = "../results/raw/";
std::string OUTPUT_FILE = "results.csv";
bool OVERWRITE_FILE = true;
int NUM_ITERATIONS = 10;

std::unordered_set<std::string> selected_cpu_benchmarks;
std::unordered_set<std::string> selected_gpu_benchmarks;
std::unordered_set<XorecVersion> selected_xorec_versions;

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
 * @brief Prints usage information and exits the program
 */
static void usage() {
  print_usage();
  print_options();
  exit(0);
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
    { "cpu-alg",        required_argument,  nullptr, 'c'  },
    { "gpu-ag",         required_argument,  nullptr, 'g'  },
    { "simd",           required_argument,  nullptr, 's'  },
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
        OUTPUT_FILE = std::string(optarg);
        if (OUTPUT_FILE.find(".csv") == std::string::npos) {
          std::cerr << "Error: Output file must have .csv extension.\n";
          exit(0);
        }
        break;
      case 'a':
        OVERWRITE_FILE = false;
        break;
      case 'c':
        args = get_arg_vector(std::string(optarg));
        for (const auto& arg: args) {
          if (CPU_BM_FUNCTIONS.find(arg) == CPU_BM_FUNCTIONS.end()) {
            std::cerr << "Error: Invalid CPU algorithm: " << arg << '\n';
            usage();
          }
          selected_cpu_benchmarks.insert(arg);
        }
        break;
      case 'g':
        args = get_arg_vector(std::string(optarg));
        for (const auto& arg: args) {
          if (GPU_BM_FUNCTIONS.find(arg) == GPU_BM_FUNCTIONS.end()) {
            std::cerr << "Error: Invalid GPU algorithm: " << arg << '\n';
            usage();
          }
          selected_gpu_benchmarks.insert(arg);
        }
        break;
      case 's':
        args = get_arg_vector(std::string(optarg));
        for (const auto& arg: args) {
          if (XOREC_VERSIONS.find(arg) == XOREC_VERSIONS.end()) {
            std::cerr << "Error: Invalid SIMD version: " << arg << '\n';
            usage();
          }
          selected_xorec_versions.insert(XOREC_VERSIONS.at(arg));
        }
        break;
      default:
        usage();
        break;
    }
  }
  if (selected_cpu_benchmarks.empty() && selected_gpu_benchmarks.empty()) {
    std::cerr << "Error: No benchmarks selected. Use --cpu-alg or --gpu-alg to select benchmarks.\n";
    usage();
  }

  if (selected_xorec_versions.empty()) {
    for (auto& [_, version] : XOREC_VERSIONS) {
      selected_xorec_versions.insert(version);
    }
  }
}

static void get_cpu_configs(std::vector<BenchmarkConfig>& configs) {
  for (const auto& block_size : VAR_BLOCK_SIZES) {
    for (const auto& ec_params : VAR_EC_PARAMS) {
      const auto& [tot_blocks, data_blocks] = ec_params;
      for (const auto& lost_blocks : VAR_NUM_LOST_BLOCKS) {
        if (lost_blocks > tot_blocks-data_blocks) continue;
        configs.push_back({
          .data_size = block_size * data_blocks,
          .block_size = block_size,
          .ec_params = ec_params,
          .num_lost_blocks = lost_blocks,
          .num_iterations = NUM_ITERATIONS,
          .gpu_computation = false,
        });
      }
    } 
  }
}

static void get_gpu_configs(std::vector<BenchmarkConfig>& configs) {
  for (const auto& block_size : VAR_BLOCK_SIZES) {
    for (const auto& ec_params : VAR_EC_PARAMS) {
      const auto& [tot_blocks, data_blocks] = ec_params;
      for (const auto& lost_blocks : VAR_NUM_LOST_BLOCKS) {
        if (lost_blocks > tot_blocks-data_blocks) continue;
        for (const auto& num_gpu_blocks : VAR_NUM_GPU_BLOCKS) {
          for (const auto& threads_per_block : VAR_NUM_THREADS_PER_BLOCK) {
            configs.push_back({
              .data_size = block_size * data_blocks,
              .block_size = block_size,
              .ec_params = ec_params,
              .num_lost_blocks = lost_blocks,
              .num_iterations = NUM_ITERATIONS,
              .gpu_computation = true,
              .num_gpu_blocks = num_gpu_blocks,
              .threads_per_gpu_block = threads_per_block,
            });
          }
        }
      }
    }
  }
}


std::string get_benchmark_name(const std::string& inp_name) {
  if (GPU_BM_NAMES.find(inp_name) != GPU_BM_NAMES.end()) return GPU_BM_NAMES.at(inp_name);
  return CPU_BM_NAMES.at(inp_name);
}

void get_benchmarks(std::vector<BenchmarkTuple>& benchmarks) {
  if (benchmarks.size() > 0) throw_error("Error: Benchmarks vector should be empty.");

  std::vector<BenchmarkConfig> cpu_configs;
  std::vector<BenchmarkConfig> gpu_configs;
  get_cpu_configs(cpu_configs);
  get_gpu_configs(gpu_configs);


  for (auto& inp_name : selected_cpu_benchmarks) {
    auto bm_name = get_benchmark_name(inp_name);
    auto bm_func = CPU_BM_FUNCTIONS.at(inp_name);

    if (inp_name.find("xorec") != std::string::npos) {
      for (const auto version : selected_xorec_versions) {
        auto full_bm_name = bm_name + ", " + get_version_name(version);
        for (auto config : cpu_configs) {
          config.xorec_version = version;
          benchmarks.push_back({ full_bm_name, bm_func, config });
        }
      }
    } else {
      for (auto config : cpu_configs) {
        benchmarks.push_back({ bm_name, bm_func, config });
      }
    }
  }

  for (auto& inp_name : selected_gpu_benchmarks) {
    auto bm_name = get_benchmark_name(inp_name);
    auto bm_func = GPU_BM_FUNCTIONS.at(inp_name);
    for (auto config : gpu_configs) {
      benchmarks.push_back({ bm_name, bm_func, config });
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
  std::unique_ptr<CSVReporter> csv_reporter = std::make_unique<CSVReporter>(RAW_DIR + OUTPUT_FILE, OVERWRITE_FILE);

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