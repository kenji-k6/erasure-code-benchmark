#include "generic_benchmark_runner.h"
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


static void BM_Baseline(benchmark::State& state) {
  BM_generic<BaselineBenchmark>(state);
}

static void BM_CM256(benchmark::State& state) {
  BM_generic<CM256Benchmark>(state);
}

static void BM_ISAL(benchmark::State& state) {
  BM_generic<ISALBenchmark>(state);
}

static void BM_Leopard(benchmark::State& state) {
  BM_generic<LeopardBenchmark>(state);
}

static void BM_Wirehair(benchmark::State& state) {
  BM_generic<WirehairBenchmark>(state);
}

BenchmarkConfig benchmark_config;
std::vector<uint32_t> lost_block_idxs = {};
const std::unordered_map<std::string, void(*)(benchmark::State&)> available_benchmarks = {
  { "baseline", BM_Baseline },
  { "cm256", BM_CM256 },
  { "isal", BM_ISAL },
  { "leopard", BM_Leopard },
  { "wirehair", BM_Wirehair }
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


void check_args(size_t s, size_t b, size_t l, double r, int i, size_t num_orig_blocks, size_t num_rec_blocks) {
  if (num_orig_blocks < 1 || num_rec_blocks < 0) {
    std::cerr << "Error: Number of original blocks must be at least 1 and number of recovery blocks must be at least 0.\n"
              << "(#original blocks) = " << num_orig_blocks << ", (#recovery blocks) = " << num_rec_blocks << "\n";
    exit(0);
  }

  // General checks
  if (s < 1) {
    std::cerr << "Error: Total data size must be greater than 0. (-s)\n";
    exit(0);
  }

  if (b < 1 || b > s) {
    std::cerr << "Error: Block size must be greater than 0 and less than the total data size. (-b)\n";
    exit(0);
  }

  if (l < 0 || l > num_rec_blocks) {
    std::cerr << "Error: Number of lost blocks must be between 0 and the number of recovery blocks. (-l)\n";
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

  // TODO: Baseline Checks
  
  // CM256 Checks
  if (selected_benchmarks.contains("cm256")) {
    if (num_orig_blocks + num_rec_blocks > ECCLimits::CM256_MAX_TOT_BLOCKS) {
      std::cerr << "Error: Total number of blocks exceeds the maximum allowed by CM256.\n"
                << "Condition: #(original blocks) + #(recovery blocks) <= " << ECCLimits::CM256_MAX_TOT_BLOCKS << "\n"
                << "(#original blocks) = " << num_orig_blocks << ", (#recovery blocks) = " << num_rec_blocks << "\n";
      exit(0);
    }
  }

  // ISA-L Checks
  if (selected_benchmarks.contains("isal")) {
    if (num_orig_blocks + num_rec_blocks > ECCLimits::ISAL_MAX_TOT_BLOCKS) {
      std::cerr << "Error: Total number of blocks exceeds the maximum allowed by ISA-L.\n"
                << "Condition: #(original blocks) + #(recovery blocks) <= " << ECCLimits::ISAL_MAX_TOT_BLOCKS << "\n"
                << "(#original blocks) = " << num_orig_blocks << ", (#recovery blocks) = " << num_rec_blocks << "\n";
      exit(0);
    }
  }


  // Leopard Checks
  if (selected_benchmarks.contains("leopard")) {
    if (num_orig_blocks + num_rec_blocks > ECCLimits::LEOPARD_MAX_TOT_BLOCKS) {
      std::cerr << "Error: Total number of blocks exceeds the maximum allowed by Leopard.\n"
                << "Condition: #(original blocks) + #(recovery blocks) <= " << ECCLimits::LEOPARD_MAX_TOT_BLOCKS << "\n"
                << "(#original blocks) = " << num_orig_blocks << ", (#recovery blocks) = " << num_rec_blocks << "\n";
      exit(0);
    }

    if (num_rec_blocks > num_orig_blocks) {
      std::cerr << "Error: Number of recovery blocks can't exceed the number of original blocks for Leopard.\n"
                << "Condition: #(recovery blocks) <= #(original blocks)\n"
                << "(#recovery blocks) = " << num_rec_blocks << ", (#original blocks) = " << num_orig_blocks << "\n";
      exit(0);
    }

    if (b % ECCLimits::LEOPARD_BLOCK_ALIGNMENT != 0) {
      std::cerr << "Error: Block size must be a multiple of " << ECCLimits::LEOPARD_BLOCK_ALIGNMENT << " for Leopard.\n";
      exit(0);
    } 
  }

  // Wirehair Checks
  if (selected_benchmarks.contains("wirehair")) {
    if (num_orig_blocks < ECCLimits::WIREHAIR_MIN_DATA_BLOCKS || num_orig_blocks > ECCLimits::WIREHAIR_MAX_DATA_BLOCKS) {
      std::cerr << "Error: Number of original blocks must be between " << ECCLimits::WIREHAIR_MIN_DATA_BLOCKS << " and " << ECCLimits::WIREHAIR_MAX_DATA_BLOCKS << " for Wirehair.\n"
                << "(#original blocks) = " << num_orig_blocks << "\n";
      exit(0);
    }
  }
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
  int option_index = 0;
  while ((c = getopt_long(argc, argv, "hs:b:l:r:i:", long_options, &option_index)) != -1) {
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
        if (long_options[option_index].name) {
          selected_benchmarks.insert(long_options[option_index].name);
        }
        break;

      default:
        usage();
        return -1;
    }
  }

  // If no benchmarks are specified, run all benchmarks
  if (selected_benchmarks.empty()) {
    for (const auto& [name, func] : available_benchmarks) {
      selected_benchmarks.insert(name);
    }
  }

  // Compute the number of original and recovery blocks
  size_t num_orig_blocks = (s + (b - 1)) / b;
  size_t num_rec_blocks = static_cast<size_t>(std::ceil(num_orig_blocks * r));

  // Check validity of passed arguments
  check_args(s, b, l, r, i, num_orig_blocks, num_rec_blocks);

  // Update the benchmark configuration
  benchmark_config.data_size = s;
  benchmark_config.block_size = b;
  benchmark_config.num_lost_blocks = l;
  benchmark_config.redundancy_ratio = r;
  benchmark_config.num_iterations = i;
  benchmark_config.computed.num_original_blocks = num_orig_blocks;
  benchmark_config.computed.num_recovery_blocks = num_rec_blocks;

  // Generate lost block indices
  select_lost_block_idxs(
    l,
    num_orig_blocks + num_rec_blocks,
    lost_block_idxs
  );
  return 0;
}



int main(int argc, char** argv) {
  // Parse command line arguments
  if (parse_args(argc, argv)) exit(0);


  // Default argument if no arguments are passed
  char arg0_default[] = "benchmark";  
  char* args_default = arg0_default;
  argc = 1;
  argv = &args_default;

  // Register benchmarks


  if (selected_benchmarks.contains("baseline")) {
    BENCHMARK(BM_Baseline)->Iterations(benchmark_config.num_iterations);
  }
  if (selected_benchmarks.contains("cm256")) {
    BENCHMARK(BM_CM256)->Iterations(benchmark_config.num_iterations);
  }
  if (selected_benchmarks.contains("isal")) {
    BENCHMARK(BM_ISAL)->Iterations(benchmark_config.num_iterations);
  }
  if (selected_benchmarks.contains("leopard")) {
    BENCHMARK(BM_Leopard)->Iterations(benchmark_config.num_iterations);
  }
  if (selected_benchmarks.contains("wirehair")) {
    BENCHMARK(BM_Wirehair)->Iterations(benchmark_config.num_iterations);
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