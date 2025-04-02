#include "bm_utils.hpp"
#include "utils.hpp"
#include "bm_functions.hpp"
#include "benchmark/benchmark.h"
#include <getopt.h>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <filesystem>

bool OVERWRITE_FILE = true;
constexpr const char* RAW_DIR = "../results/raw/";
std::string OUTPUT_FILE = "results.csv";
int NUM_ITERATIONS = 1;

std::chrono::system_clock::time_point START_TIME;
using BenchmarkTuple = std::tuple<std::string, BenchmarkFunction, BenchmarkConfig>;

std::unordered_set<std::string> cpu_selected_algorithms;
std::unordered_set<std::string> gpu_selected_algorithms;

const std::unordered_map<std::string, BenchmarkFunction> CPU_BENCHMARK_FUNCTIONS = {
  { "cm256", BM_CM256 },
  { "isal", BM_ISAL },
  { "leopard", BM_Leopard},
  { "wirehair", BM_Wirehair },
  { "xorec", BM_XOREC }
};

const std::unordered_map<std::string, std::string> CPU_BENCHMARK_NAMES = {
  { "cm256", "CM256 (MDS)" },
  { "isal", "ISA-L (MDS)" },
  { "leopard", "Leopard (MDS)"},
  { "wirehair", "Wirehair (MDS)" },
  { "xorec", "Xorec" }
};

const std::unordered_map<std::string, std::string> GPU_BENCHMARK_NAMES = {
  { "xorec", "Xorec GPU"                         },
  { "xorec-cpu-parity", "Xorec GPU (CPU Parity)" }
};


const std::unordered_map<std::string, BenchmarkFunction> GPU_BENCHMARK_FUNCTIONS = {
  { "xorec", BM_XOREC_GPU }, 
  { "xorec-cpu-parity", BM_XOREC_GPU_PARITY_CPU }
};



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


void parse_args(int argc, char** argv) {
  struct option long_options[] = {
    { "help",           no_argument,        nullptr, 'h' },
    { "file",           required_argument,  nullptr, 'f' },
    { "append",         no_argument,        nullptr, 'a' },
    { "iterations",     required_argument,  nullptr, 'i' },
    { "cpu-algorithm",  required_argument,  nullptr, 'c' },
    { "gpu-algorithm",  required_argument,  nullptr, 'g' },
    { nullptr,          0,                  nullptr,  0  }
  };

  int c;
  int option_index = 0;
  std::string flag;
  std::vector<std::string> args;

  while ((c = getopt_long(argc, argv, "hf:ai:c:g:", long_options, &option_index)) != -1) {
    switch (c) {
      case 'h':
        print_usage();
        exit(0);
        break;
      case 'f':
        OUTPUT_FILE = std::string(optarg);
        break;
      case 'a':
        OVERWRITE_FILE = false;
        break;
      case 'i':
        NUM_ITERATIONS = std::stoi(optarg);
        if (NUM_ITERATIONS < 3) throw_error("Number of iterations must be at least 3");
        break;
      case 'c':
        args = get_arg_vector(std::string(optarg));
        for (const auto& arg : args) {
          if (CPU_BENCHMARK_FUNCTIONS.find(arg) == CPU_BENCHMARK_FUNCTIONS.end()) throw_error("Invalid CPU algorithm: " + arg);
          cpu_selected_algorithms.insert(arg);
        }
        break;
      case 'g':
        args = get_arg_vector(std::string(optarg));
        for (const auto& arg : args) {
          if (GPU_BENCHMARK_FUNCTIONS.find(arg) == GPU_BENCHMARK_FUNCTIONS.end()) throw_error("Invalid GPU algorithm: " + arg);
          gpu_selected_algorithms.insert(arg);
        }
        break;
      default:
        print_usage();
        exit(0);
    }
  }
}

void print_usage() {
  std::cout << '\n' << "Usage: " << "ec-benchmark [OPTIONS(0)...] [ : [OPTIONS(N)...]]" << '\n' << '\n'
            << "Options:" << '\n'
            << "  -h, --help            show this help message"                                             << '\n'
            << "  -f, --file            specify the output CSV file (inside /results/raw/)"                 << '\n'
            << "  -a, --append          append results to the output file (default: overwrite)"             << '\n'
            << "  -i, --iterations      number of benchmark iterations (atleast 3, default 3)"                         << '\n'
            << "  -c, --cpu-algorithm <cm256|isal|leopard|wirehair|xorec>"                                  << '\n'
            << "                        run the specified CPU algorithms, 0 or more comma separated args."  << '\n'
            << "  -g, --gpu-algorithm <xorec,xorec-cpu-parity>"                                                              << '\n'
            << "                        run the specified GPU algorithms, 0 or more comma separated args."  << '\n' << '\n';
}



void get_cpu_benchmarks(std::vector<BenchmarkTuple>& benchmarks) {
  for (auto block_size : VAR_BLOCK_SIZES) {
    for (auto fec_params : VAR_FEC_PARAMS) {
      BenchmarkConfig config {
        FIXED_MESSAGE_SIZE,
        block_size,
        fec_params,
        FIXED_NUM_LOST_RDMA_PKTS,
        false,
        0,
        0,
        NUM_ITERATIONS,
        nullptr
      };

      for (auto alg : cpu_selected_algorithms) {
        auto name = CPU_BENCHMARK_NAMES.at(alg);
        auto func = CPU_BENCHMARK_FUNCTIONS.at(alg);
        benchmarks.push_back({ name, func, config });
      }
    }
  }
}

void get_gpu_benchmarks(std::vector<BenchmarkTuple>& benchmarks) {
  for (auto block_size : VAR_BLOCK_SIZES) {
    for (auto fec_params : VAR_FEC_PARAMS) {
      for (auto num_gpu_blocks : VAR_NUM_GPU_BLOCKS) {
        for (auto threads_per_gpu_block : VAR_THREADS_PER_GPU_BLOCK) {
          BenchmarkConfig config {
            FIXED_MESSAGE_SIZE,
            block_size,
            fec_params,
            FIXED_NUM_LOST_RDMA_PKTS,
            true,
            num_gpu_blocks,
            threads_per_gpu_block,
            NUM_ITERATIONS,
            nullptr
          };

          for (auto alg : gpu_selected_algorithms) {
            auto name = GPU_BENCHMARK_NAMES.at(alg);
            auto func = GPU_BENCHMARK_FUNCTIONS.at(alg);
            benchmarks.push_back({ name, func, config });
          }
        }
      }
    }
  }
}

static void ensure_result_dir() {
  if (!std::filesystem::exists(RAW_DIR)) {
    std::filesystem::create_directory(RAW_DIR);
  }
}

void run_benchmarks() {
  START_TIME = std::chrono::system_clock::now();
  ensure_result_dir();

  std::string repetitions = "--benchmark_repetitions=" + std::to_string(5);
  char arg0[] = "benchmark";
  char arg1[] = "--benchmark_out=console";
  char arg2[50];
  snprintf(arg2, sizeof(arg2), "--benchmark_repetitions=%d", NUM_ITERATIONS);

  int argc = 3;
  char *argv[] = { arg0, arg1, arg2 };

  std::vector<BenchmarkTuple> benchmarks;

  get_cpu_benchmarks(benchmarks);
  get_gpu_benchmarks(benchmarks);

  if (benchmarks.empty()) {
    throw_error("No benchmarks selected. Use -c or -g to select benchmarks.");
  }

  std::unique_ptr<BenchmarkProgressReporter> console_reporter = std::make_unique<BenchmarkProgressReporter>(benchmarks.size() * NUM_ITERATIONS, START_TIME);
  std::unique_ptr<BenchmarkCSVReporter> csv_reporter = std::make_unique<BenchmarkCSVReporter>(RAW_DIR + OUTPUT_FILE, OVERWRITE_FILE, NUM_ITERATIONS);

  benchmark::ClearRegisteredBenchmarks();
  for (auto [name, func, cfg] : benchmarks) {
    cfg.progress_reporter = console_reporter.get();

    benchmark::RegisterBenchmark(name, func, cfg)
      ->Iterations(1)
      ->DisplayAggregatesOnly(true)
      ->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
        return *(std::min_element(v.begin(), v.end()));
      })
      ->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
        return *(std::max_element(v.begin(), v.end()));
      });
  }

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return;
  benchmark::RunSpecifiedBenchmarks(console_reporter.get(), csv_reporter.get());
  benchmark::Shutdown();
}