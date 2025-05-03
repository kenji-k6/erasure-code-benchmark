/**
 * @file benchmark_suite.cpp
 * @brief Internal implementations for parsing arguments, constructing benchmarks, and running them.
 */

#include "benchmark_cli.hpp"
#include "benchmark_suite.hpp"
#include "benchmark/benchmark.h"
#include "console_reporter.hpp"
#include "csv_reporter.hpp"
#include "runners.hpp"
#include "utils.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <getopt.h>
#include <ranges>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace {

  /// Internal configuration & globals
  constexpr std::string_view RAW_DIR = "../results/raw/";
  inline std::string OUTPUT_FILE = "results.csv";
  inline bool OVERWRITE_FILE = true;
  inline int NUM_ITERATIONS = 10;
  inline int NUM_WARMUP_ITERATIONS = 0;

  std::unordered_set<std::string> selected_cpu_benchmarks;
  std::unordered_set<std::string> selected_gpu_benchmarks;
  std::unordered_set<XorecVersion> selected_xorec_versions;

  /// Function maps and names
  inline const std::unordered_map<std::string, BenchmarkFunction> CPU_BM_FUNCTIONS = {
    { "cm256",              BM_CM256              },
    { "isal",               BM_ISAL               },
    { "leopard",            BM_Leopard            },
    { "xorec",              BM_XOREC              },
    { "xorec-unified-ptr",  BM_XOREC_UNIFIED_PTR  },
    { "xorec-gpu-ptr",      BM_XOREC_GPU_PTR      }
  };
  
  const std::unordered_map<std::string, std::string> CPU_BM_NAMES = {
    { "cm256",              "CM256"                   },
    { "isal",               "ISA-L"                   },
    { "leopard",            "Leopard"                 },
    { "xorec",              "XOR-EC"                  },
    { "xorec-unified-ptr",  "XOR-EC (Unified Memory)" },
    { "xorec-gpu-ptr",      "XOR-EC (GPU Memory)"     }
  };

  const std::unordered_map<std::string, BenchmarkFunction> GPU_BM_FUNCTIONS = {
    { "xorec-gpu",      BM_XOREC_GPU_CMP }
  };
  
  const std::unordered_map<std::string, std::string> GPU_BM_NAMES = {
    { "xorec-gpu",  "XOR-EC (GPU Computation)" }
  };
  
  const std::unordered_map<std::string, XorecVersion> XOREC_VERSIONS = {
    { "scalar",  XorecVersion::Scalar  },
    { "sse2",    XorecVersion::SSE2    },
    { "avx2",    XorecVersion::AVX2    },
    { "avx512",  XorecVersion::AVX512  }
  };

  /**
   * @brief Split comma-separated string using ranges and transform
   * each substring to lower case
   * @param input User's command line input
   * @return std::vector<std::string> 
   */
  std::vector<std::string> get_arg_vector(std::string_view input) {
    auto split_view = std::views::split(input, ',');
    auto tokens_view = std::views::transform([](auto&& subrange) {
      // Construct a string formt the subrange and convert to lower case
      std::string_view sv(&*std::ranges::begin(subrange), std::ranges::distance(subrange));
      return to_lower(std::string(sv));
    })(split_view);
    return std::vector<std::string>(tokens_view.begin(), tokens_view.end());
  }

  /**
   * @brief Print usage information and exit
   */
  [[noreturn]] void usage() {
    print_usage();
    print_options();
    std::exit(EXIT_SUCCESS);
  }

  /**
   * @brief Parses the command-line arguments and sets the global variables
   * 
   * @param argc 
   * @param argv 
   */
  void parse_args(int argc, char** argv) {
    struct option long_options[] = {
      { "help",       no_argument,        nullptr, 'h'  },
      { "file",       required_argument,  nullptr, 'f'  },
      { "append",     no_argument,        nullptr, 'a'  },
      { "iterations", required_argument,  nullptr, 'i'  },
      { "warmup",     required_argument,  nullptr, 'w'  },
      { "cpu",        required_argument,  nullptr, 'c'  },
      { "gpu",        required_argument,  nullptr, 'g'  },
      { "simd",       required_argument,  nullptr, 's'  },
      { nullptr,      0,                  nullptr,  0   }
    };

    int c;
    int option_index = 0;
    std::vector<std::string> args;

    while((c = getopt_long(argc, argv, "hf:ai:w:c:g:s:", long_options, &option_index)) != -1) {
      switch (c) {
        case 'h': {
          usage();
          break;
        }

        case 'f': {
          OUTPUT_FILE = optarg;
          if (OUTPUT_FILE.find(".csv") == std::string::npos) {
            std::cerr << "Error: Output file must have .csv extension.\n";
            std::exit(EXIT_FAILURE);
          }
          break;
        }

        case 'a': {
          OVERWRITE_FILE = false;
          break;
        }

        case 'i': {
          NUM_ITERATIONS = std::stoi(optarg);
          if (NUM_ITERATIONS <= 0) {
            std::cerr << "Error: Number of iterations must be positive.\n";
            std::exit(EXIT_FAILURE);
          }
          break;
        }

        case 'w': {
          NUM_WARMUP_ITERATIONS = std::stoi(optarg);
          if (NUM_WARMUP_ITERATIONS < 0) {
            std::cerr << "Error: Number of warmup iterations must be non-negative.\n";
            std::exit(EXIT_FAILURE);
          }
          break;
        }

        case 'c': {
          args = get_arg_vector(optarg);
          for (const auto& arg : args) {
            if (CPU_BM_FUNCTIONS.find(arg) == CPU_BM_FUNCTIONS.end()) {
              std::cerr << "Error: Invalid CPU algorithm: " << arg << '\n';
              std::exit(EXIT_FAILURE);
            }
            selected_cpu_benchmarks.insert(arg);
          }
          break;
        }

        case 'g': {
          args = get_arg_vector(optarg);
          for (const auto& arg : args) {
            if (GPU_BM_FUNCTIONS.find(arg) == GPU_BM_FUNCTIONS.end()) {
              std::cerr << "Error: Invalid GPU algorithm: " << arg << '\n';
              std::exit(EXIT_FAILURE);
            }
            selected_gpu_benchmarks.insert(arg);
          }
          break;
        }

        case 's': {
          args = get_arg_vector(optarg);
          for (const auto& arg : args) {
            if (XOREC_VERSIONS.find(arg) == XOREC_VERSIONS.end()) {
              std::cerr << "Error: Invalid SIMD version: " << arg << '\n';
              std::exit(EXIT_FAILURE);
            }
            selected_xorec_versions.insert(XOREC_VERSIONS.at(arg));
          }
          break;
        }

        default: {
          usage();
          break;
        }
      }
    }

    if (selected_cpu_benchmarks.empty() && selected_gpu_benchmarks.empty()) {
      std::cerr << "Error: No benchmarks selected. Use --cpu-alg or --gpu-alg to select benchmarks.\n";
      std::exit(EXIT_FAILURE);
    }

    // If SIMD versions wre not specified, default to all versions.
    if (selected_xorec_versions.empty()) {
      for (const auto& [_, version] : XOREC_VERSIONS) {
        selected_xorec_versions.insert(version);
      }
    }
  }

  /**
   * @brief Populate the benchmarks vector with all the benchmarks
   * based on the selected algorithms and configurations.
   * 
   * @param benchmarks empty vector to be populated with benchmarks
   */
  void get_benchmarks(std::vector<BenchmarkTuple>& benchmarks) {
    if (!benchmarks.empty()) {
      std::cerr << "Error: Benchmarks already generated. Cannot generate again.\n";
      std::exit(EXIT_FAILURE);
    }
    std::vector<BenchmarkConfig> cpu_configs;
    std::vector<BenchmarkConfig> gpu_configs;

    auto get_cpu_configs = [&cpu_configs]() {
      for (const auto& data_size : VAR_DATA_SIZES) {
        for (const auto& ec_params : VAR_EC_PARAMS) {
          const auto& [tot_blocks, data_blocks] = ec_params;
          const auto block_size = data_size / data_blocks;
          for (const auto& lost_blocks : VAR_NUM_LOST_BLOCKS) {
            if (lost_blocks > tot_blocks-data_blocks) continue;
            cpu_configs.push_back({
              .data_size = data_size,
              .block_size = block_size,
              .ec_params = ec_params,
              .num_lost_blocks = lost_blocks,
              .num_iterations = NUM_ITERATIONS,
              .num_warmup_iterations = NUM_WARMUP_ITERATIONS,
              .gpu_computation = false,
            });
          }
        } 
      }
    };

    auto get_gpu_configs = [&gpu_configs]() {
      for (const auto& data_size : VAR_DATA_SIZES) {
        for (const auto& ec_params : VAR_EC_PARAMS) {
          const auto& [tot_blocks, data_blocks] = ec_params;
          const auto block_size = data_size / data_blocks;
          for (const auto& lost_blocks : VAR_NUM_LOST_BLOCKS) {
            if (lost_blocks > tot_blocks-data_blocks) continue;
            for (const auto& num_gpu_blocks : VAR_NUM_GPU_BLOCKS) {
              for (const auto& threads_per_block : VAR_NUM_THREADS_PER_BLOCK) {
                gpu_configs.push_back({
                  .data_size = data_size,
                  .block_size = block_size,
                  .ec_params = ec_params,
                  .num_lost_blocks = lost_blocks,
                  .num_iterations = NUM_ITERATIONS,
                  .num_warmup_iterations = NUM_WARMUP_ITERATIONS,
                  .gpu_computation = true,
                  .num_gpu_blocks = num_gpu_blocks,
                  .threads_per_gpu_block = threads_per_block,
                });
              }
            }
          }
        }
      }
    };

    auto get_benchmark_name = [](const std::string& input_name) -> std::string {
      if (GPU_BM_NAMES.find(input_name) != GPU_BM_NAMES.end()) {
        return GPU_BM_NAMES.at(input_name);
      }
      return CPU_BM_NAMES.at(input_name);
    };


    get_cpu_configs();
    get_gpu_configs();

    // CPU Benchmarks
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
    
    // GPU Benchmarks
    for (auto& inp_name : selected_gpu_benchmarks) {
      auto bm_name = get_benchmark_name(inp_name);
      auto bm_func = GPU_BM_FUNCTIONS.at(inp_name);
      for (auto config : gpu_configs) {
        benchmarks.push_back({ bm_name, bm_func, config });
      }
    }
  }

} // namespace


/**
 * @brief Parse the command-line arguments, construct the benchmarks and run them
 * 
 * @param argc 
 * @param argv 
 */
void run_benchmarks(int argc, char** argv) {
  auto start_time = std::chrono::system_clock::now();

  // Prepare new argument list for benchmark::Initialize
  int new_argc = 2;
  char *new_argv[] = { (char*)"benchmark", (char*)"--benchmark_out=console" };

  // Parse command-line arguments
  parse_args(argc, argv);

  // Construct the benchmarks
  std::vector<BenchmarkTuple> benchmarks;
  get_benchmarks(benchmarks);
  // Create reporters
  auto console_reporter = std::make_unique<ConsoleReporter>(
    NUM_ITERATIONS * benchmarks.size(), start_time);
  auto csv_reporter = std::make_unique<CSVReporter>(
    std::string(RAW_DIR) + OUTPUT_FILE, OVERWRITE_FILE);


  benchmark::ClearRegisteredBenchmarks();

  // Register each benchmark
  for (const auto& [name, func, cfg] : benchmarks) {
    auto config = cfg;
    config.reporter = console_reporter.get();
    benchmark::RegisterBenchmark(name, func, config)
      ->UseRealTime()
      ->Iterations(NUM_ITERATIONS);
  }

  benchmark::Initialize(&new_argc, new_argv);
  if (benchmark::ReportUnrecognizedArguments(new_argc, new_argv)) return;
  benchmark::RunSpecifiedBenchmarks(console_reporter.get(), csv_reporter.get());
  benchmark::Shutdown();
}