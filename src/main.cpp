#include "benchmark_runner.h"
#include "leopard_benchmark.h"
#include "cm256_benchmark.h"
#include "wirehair_benchmark.h"
#include "isal_benchmark.h"
#include "baseline_benchmark.h"
#include "benchmark/benchmark.h"
#include "utils.h"

#include <memory>
#include <iostream>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <getopt.h>

BenchmarkConfig benchmark_config;
std::vector<uint32_t> lost_block_idxs = {};
const std::unordered_map<std::string, void(*)(benchmark::State&)> available_benchmarks = {
  { "baseline", BM_generic<BaselineBenchmark>> },
  { "cm256", BM_generic<CM256Benchmark> },
  { "isal", BM_generic<ISALBenchmark> },
  { "leopard", BM_generic<LeopardBenchmark> },
  { "wirehair", BM_generic<WirehairBenchmark> }
};

std::unordered_set<std::string> selected_benchmarks;


void usage() {
  std::cerr << "Usage: ecc-benchmark [options]\n"
            << "  -h | --help       Help\n"
            << "  -s <size>         Total size of original data in bytes (default 520'192B)\n"
            << "  -b <size>         Size of each block in bytes (default 4'096B)\n"
            << "  -l <num>          Number of lost blocks (default 64)\n"
            << "  -r <ratio>        Redundancy ratio (#recovery blocks / #original blocks) (default 1.0)\n"
            << "  -i <num>          Number of benchmark iterations (default 1)\n\n"

            << " The following flags are used to specify which benchmarks to run. \n"
            << " If no flags are specified, all benchmarks will be run.\n"
            << "  --baseline        Run the baseline benchmark\n"
            << "  --cm256           Run the CM256 benchmark\n"
            << "  --isal            Run the ISA-L benchmark\n"
            << "  --leopard         Run the Leopard benchmark\n"
            << "  --wirehair        Run the Wirehair benchmark\n";
  exit(0);
}


void check_args(size_t s, size_t b, size_t l, double r, int i) {
  size_t num_orig_blocks = (s + (b - 1)) / b;
  size_t num_rec_blocks = static_cast<size_t>(std::ceil(num_orig_blocks * r));
  //TODO: check that lost_blocks <= recovery_blocks (maybe)

  // General checks
  if (s < 1) {
    std::cerr << "Error: Total data size must be greater than 0. (-s)\n";
    exit(0);
  }

  if (b < 1 || b > s) {
    std::cerr << "Error: Block size must be greater than 0 and less than the total data size. (-b)\n";
    exit(0);
  }

  if (l < 0 || l > num_orig_blocks + num_rec_blocks) {
    std::cerr << "Error: Number of lost blocks must be between 0 and the total number of blocks. (-l)\n";
    exit(0);
  }

  if (r < 0) {
    std::cerr << "Error: Redundancy ratio must be at least 0. (-r)\n";
    exit(0);
  }

  if (i < 1) {
    std::cerr << "Error: Number of iterations must be at least 1. (-i)\n";
    exit(0);
  }

  // Library Specific checks

  // TODO: Baseline Checks
  
  // CM256 Checks
  if (selected_benchmarks.contains("cm256")) {
    if (num_orig_blocks + num_rec_blocks > ECCLimits::CM256_MAX_TOT_BLOCKS) {
      std::cerr << "Error: Total number of blocks exceeds the maximum allowed by CM256.\n"
                << "Condition: #(original blocks) + #(recovery blocks) <= " << ECCLimits::CM256_MAX_TOT_BLOCKS << "\n"
                << "(#original blocks) = " << num_orig_blocks << ", (#recovery blocks) = " << num_rec_blocks << "\n";
    }
    exit(0);
  }

  // ISA-L Checks
  if (selected_benchmarks.contains("isal")) {
    if (num_orig_blocks + num_rec_blocks > ECCLimits::ISAL_MAX_TOT_BLOCKS) {
      std::cerr << "Error: Total number of blocks exceeds the maximum allowed by ISA-L.\n"
                << "Condition: #(original blocks) + #(recovery blocks) <= " << ECCLimits::ISAL_MAX_TOT_BLOCKS << "\n"
                << "(#original blocks) = " << num_orig_blocks << ", (#recovery blocks) = " << num_rec_blocks << "\n";
    }

    if (l > num_rec_blocks) {
      std::cerr << "Error: Number of lost blocks can't exceed the number of recovery blocks for ISA-L.\n"
                << "Condition: #(lost blocks) <= #(recovery blocks)\n"
                << "(#lost blocks) = " << l << ", (#recovery blocks) = " << num_rec_blocks << "\n";
    }
  }


  // 

} 



int parse_args(int argc, char** argv) {
  size_t s = 520'192;
  size_t b = 4'096;
  size_t l = 64;
  double r = 1.0;
  int i = 1;

  struct option long_options[] = {
    { "help",     no_argument, nullptr, 'h' },
    { "baseline", no_argument, nullptr, 0 },
    { "cm256",    no_argument, nullptr, 0 },
    { "isal",     no_argument, nullptr, 0 },
    { "leopard",  no_argument, nullptr, 0 },
    { "wirehair", no_argument, nullptr, 0},
    { nullptr,    0,           nullptr, 0 }
  };

  int c;
  while ((c = getopt_long(argc, argv, "hs:b:l:r:i:", long_options, nullptr)) != -1) {
    switch (c) {
      case 'h':
        usage();
        break;

      case 's':
        s = std::stoull(optarg);
        break;

      case 'b':
        b = std::stoull(optarg);
        break;
      
      case 'l':
        l = std::stoull(optarg);
        break;

      case 'r':
        r = std::stod(optarg);
        break;
      
      case 'i':
        i = std::stoi(optarg);
        break;

      case 0: // Long options
        if (available_benchmarks.count(optarg)) selected_benchmarks.insert(optarg);
        break;

      default:
        usage();
        return -1;
    }
  }
  return 0;
}



static void BM_cm256(benchmark::State& state) {
  BM_generic<CM256Benchmark>(state);
}

static void BM_leopard(benchmark::State& state) {
  BM_generic<LeopardBenchmark>(state);
}

/*
 * Important: Wirehair does not accept a specified no. of recovery blocks
 * It keeps sending blocks until the decoder has enough to recover the original data
*/
static void BM_wirehair(benchmark::State& state) {
  BM_generic<WirehairBenchmark>(state);
}

static void BM_isal(benchmark::State& state) {
  BM_generic<ISALBenchmark>(state);
}

static void BM_baseline(benchmark::State& state) {
  BM_generic<BaselineBenchmark>(state);
}



// TODO: Pass arguments
BenchmarkConfig get_config() {
  BenchmarkConfig config;
  config.data_size = 81'280'000; //1073736320; // ~~1.0737 GB
  config.block_size = 640000; //6'316'096; // 6316.096 KB
  config.num_lost_blocks = 10;
  config.redundancy_ratio = 1;
  config.num_iterations = 1;
  config.computed.num_original_blocks = (config.data_size + (config.block_size - 1)) / config.block_size;
  config.computed.num_recovery_blocks = static_cast<size_t>(std::ceil(config.computed.num_original_blocks * config.redundancy_ratio));
  return config;
}

int main(int argc, char** argv) {
  std::cout << argc << std::endl;
  // Get and compute the configuration
  benchmark_config = get_config();

  // Get the lost block indexes
  select_lost_block_idxs(
    benchmark_config.num_lost_blocks,
    benchmark_config.computed.num_original_blocks + benchmark_config.computed.num_recovery_blocks,
    lost_block_idxs
  );

  // Register Benchmarks
  BENCHMARK(BM_cm256)->Iterations(benchmark_config.num_iterations);
  BENCHMARK(BM_leopard)->Iterations(benchmark_config.num_iterations);
  BENCHMARK(BM_wirehair)->Iterations(benchmark_config.num_iterations);
  BENCHMARK(BM_isal)->Iterations(benchmark_config.num_iterations);
  BENCHMARK(BM_baseline)->Iterations(benchmark_config.num_iterations);


  // Default argument if no arguments are passed
  char arg0_default[] = "benchmark";  
  char* args_default = arg0_default;

  // If no arguments are passed, set argc to 1 and argv to point to the default argument
  if (argc == 0 || argv == nullptr) {
    argc = 1;
    argv = &args_default;
  }

  // Initialize Google Benchmark
  ::benchmark::Initialize(&argc, argv);

  // Check and report unrecognized arguments
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
      return 1; // Return error code if there are unrecognized arguments
  }


  // Run all specified benchmarks
  ::benchmark::RunSpecifiedBenchmarks();

  // Shutdown Google Benchmark
  ::benchmark::Shutdown();
}