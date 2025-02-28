#include "generic_benchmark_runner.h"
#include "leopard_benchmark.h"
#include "cm256_benchmark.h"
#include "wirehair_benchmark.h"
#include "isal_benchmark.h"
#include "baseline_benchmark.h"
#include "benchmark/benchmark.h"
#include "utils.h"
#include "benchmark_result_writer.h"

#include <memory>
#include <iostream>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <getopt.h>

#define OUTPUT_FILE_PATH "../results/raw/"
#define TAKE_CMD_LINE_ARGS true
#define RUNNING_ON_DOCKER true

#if RUNNING_ON_DOCKER

  #define FIXED_NUM_ITERATIONS 50

  // Fixed buffer size of 64 MiB
  #define FIXED_BUFFER_SIZE 67108864

  // Fixed num blocks
  #define FIXED_NUM_ORIGINAL_BLOCKS 128
  #define FIXED_NUM_RECOVERY_BLOCKS 4

  #define FIXED_PARITY_RATIO 0.03125

  #define FIXED_NUM_LOST_BLOCKS 1

  // Variable buffer size(s) (1 MiB - 256 MiB)
  #define VAR_BUFFER_SIZE { 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456 }
  #define VAR_NUM_RECOVERY_BLOCKS { 1, 2, 4, 8, 16, 32, 64, 128 }
  #define VAR_NUM_LOST_BLOCKS { 1, 2, 4, 8, 16, 32, 64, 128 }


#else

  #define FIXED_NUM_ITERATIONS 2500

  // Fixed buffer size of 1 GiB
  #define FIXED_BUFFER_SIZE 1073741824

  // Fixed num blocks
  #define FIXED_NUM_ORIGINAL_BLOCKS 128
  #define FIXED_NUM_RECOVERY_BLOCKS 4

  #define FIXED_NUM_LOST_BLOCKS 1

  // Variable buffer size(s) (1 MiB -  8 GiB)
  #define VAR_BUFFER_SIZE { 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296, 8589934592 }
  #define VAR_NUM_RECOVERY_BLOCKS { 1, 2, 4, 8, 16, 32, 64, 128 }
  #define VAR_NUM_LOST_BLOCKS { 1, 2, 4, 8, 16, 32, 64, 128 }

#endif



static void BM_Baseline(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<BaselineBenchmark>(state, config);
}

static void BM_CM256(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<CM256Benchmark>(state, config);
}

static void BM_ISAL(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<ISALBenchmark>(state, config);
}

static void BM_Leopard(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<LeopardBenchmark>(state, config);
}

static void BM_Wirehair(benchmark::State& state, const BenchmarkConfig& config) {
  BM_generic<WirehairBenchmark>(state, config);
}

const std::unordered_map<std::string, void(*)(benchmark::State&, const BenchmarkConfig&)> available_benchmarks = {
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
            << "  --wirehair        Run the Wirehair benchmark\n"
            << " *If no arguments at all are specified, the full selection of benchmarks, over multiple configurations, will be run.\n";
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

  // Baseline checks
  if (selected_benchmarks.contains("baseline")) {
    if (b % ECCLimits::BASELINE_BLOCK_ALIGNMENT != 0) {
      std::cerr << "Error: Block size must be a multiple of " << ECCLimits::BASELINE_BLOCK_ALIGNMENT << " for Baseline.\n";
      exit(0);
    }
  }
  
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



BenchmarkConfig parse_args(int argc, char** argv) {
  size_t s = 520'192;
  size_t b = 4'096;
  size_t l = 64;
  double r = 1.0;
  int i = 1;

  struct option long_options[] = {
    { "help",     no_argument, nullptr, 'h' },
    // { "baseline", no_argument, nullptr, 0 },
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
        exit(0);
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

  // Initialize the benchmark configuration
  BenchmarkConfig benchmark_config;

  benchmark_config.data_size = s;
  benchmark_config.block_size = b;

  benchmark_config.num_lost_blocks = l;
  benchmark_config.redundancy_ratio = r;

  benchmark_config.computed.num_original_blocks = num_orig_blocks;
  benchmark_config.computed.num_recovery_blocks = num_rec_blocks;

  benchmark_config.num_iterations = i;
  benchmark_config.plot_id = -1;

  return benchmark_config;
}


#if TAKE_CMD_LINE_ARGS
int main(int argc, char** argv) {
  BenchmarkConfig benchmark_config = parse_args(argc, argv);

  // Initialize lost block indices
  std::vector<uint32_t> lost_block_idxs(benchmark_config.num_lost_blocks);
  select_lost_block_idxs(
    benchmark_config.num_lost_blocks,
    benchmark_config.computed.num_original_blocks + benchmark_config.computed.num_recovery_blocks,
    lost_block_idxs.data()
  );
  benchmark_config.lost_block_idxs = lost_block_idxs.data();



  char* new_args[] = {
    (char *)"benchmark",
  };

  argc = 1;
  argv = new_args;

  // Register benchmarks
  for (const auto& [name, func] : available_benchmarks) {
    if (selected_benchmarks.contains(name)) {
      benchmark::RegisterBenchmark(name.c_str(), func, benchmark_config)->Iterations(benchmark_config.num_iterations);
    }
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

#else

int main (int argc, char** argv) {


  char* new_args[] = {
    (char *)"benchmark",
    (char *)"--benchmark_out=console"
  };

  argc = 2;
  argv = new_args;


  // Initialize reporter
  std::string output_file = "benchmark_results.csv";
  BenchmarkCSVReporter reporter(OUTPUT_FILE_PATH + output_file, true);
  std::vector<BenchmarkConfig> configs;

  /// @attention Configs for varying buffer sizes
  std::vector<uint32_t>fixed_num_lost_block_idxs(FIXED_NUM_LOST_BLOCKS);

  select_lost_block_idxs(
    FIXED_NUM_LOST_BLOCKS,
    FIXED_NUM_ORIGINAL_BLOCKS + FIXED_NUM_RECOVERY_BLOCKS,
    fixed_num_lost_block_idxs.data()
  );


  for (auto buf_size : VAR_BUFFER_SIZE) {
    BenchmarkConfig config;
    config.data_size = buf_size;
    config.block_size = buf_size / FIXED_NUM_ORIGINAL_BLOCKS;

    config.num_lost_blocks = FIXED_NUM_LOST_BLOCKS;
    config.lost_block_idxs = fixed_num_lost_block_idxs.data();
    config.redundancy_ratio = FIXED_PARITY_RATIO;
    
    config.computed.num_original_blocks = FIXED_NUM_ORIGINAL_BLOCKS;
    config.computed.num_recovery_blocks = FIXED_NUM_RECOVERY_BLOCKS;

    config.num_iterations = FIXED_NUM_ITERATIONS;
    config.plot_id = 0;
    configs.push_back(config);
  }


  /// @attention Configs for varying redundancy ratio
  for (auto num_rec_blocks : VAR_NUM_RECOVERY_BLOCKS) {
    BenchmarkConfig config;
    config.data_size = FIXED_BUFFER_SIZE;
    config.block_size = FIXED_BUFFER_SIZE / FIXED_NUM_ORIGINAL_BLOCKS;

    config.num_lost_blocks = FIXED_NUM_LOST_BLOCKS;
    config.lost_block_idxs = fixed_num_lost_block_idxs.data();
    config.redundancy_ratio = static_cast<double>(num_rec_blocks) / FIXED_NUM_ORIGINAL_BLOCKS;

    config.computed.num_original_blocks = FIXED_NUM_ORIGINAL_BLOCKS;
    config.computed.num_recovery_blocks = num_rec_blocks;

    config.num_iterations = FIXED_NUM_ITERATIONS;
    config.plot_id = 1;
    configs.push_back(config);
  }


  /// @attention Configs for varying no. of lost blocks the no. of redundancy blocks is always == no. of lost blocks
  uint32_t tot_num_lost_blocks = 0;
  for (auto num_lost_blocks : VAR_NUM_LOST_BLOCKS) {
    tot_num_lost_blocks += num_lost_blocks;
  }

  std::vector<uint32_t> var_num_lost_block_idxs(tot_num_lost_blocks);

  uint32_t *curr = var_num_lost_block_idxs.data();

  for (auto num_lost_blocks : VAR_NUM_LOST_BLOCKS) {
    BenchmarkConfig config;

    select_lost_block_idxs(
      num_lost_blocks,
      FIXED_NUM_ORIGINAL_BLOCKS + num_lost_blocks,
      curr
    );

    config.data_size = FIXED_BUFFER_SIZE;
    config.block_size = FIXED_BUFFER_SIZE / FIXED_NUM_ORIGINAL_BLOCKS;

    config.num_lost_blocks = num_lost_blocks;
    config.lost_block_idxs = curr;
    config.redundancy_ratio = static_cast<double>(num_lost_blocks) / FIXED_NUM_ORIGINAL_BLOCKS;

    config.computed.num_original_blocks = FIXED_NUM_ORIGINAL_BLOCKS;
    config.computed.num_recovery_blocks = num_lost_blocks;

    config.num_iterations = FIXED_NUM_ITERATIONS;
    config.plot_id = 2;
    configs.push_back(config);

    curr += num_lost_blocks;
  }



  for (auto& config : configs) {
    benchmark::RegisterBenchmark("Baseline", BM_Baseline, config)->Iterations(config.num_iterations);
    benchmark::RegisterBenchmark("CM256", BM_CM256, config)->Iterations(config.num_iterations);
    benchmark::RegisterBenchmark("ISA-L", BM_ISAL, config)->Iterations(config.num_iterations);
    benchmark::RegisterBenchmark("Leopard", BM_Leopard, config)->Iterations(config.num_iterations);
    benchmark::RegisterBenchmark("Wirehair", BM_Wirehair, config)->Iterations(config.num_iterations);
  }


  // Initialize Google Benchmark
  benchmark::Initialize(&argc, argv);

  // Check and report unrecognized arguments
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
      return 1; // Return error code if there are unrecognized arguments
  }

  // Run all specified benchmarks
  benchmark::RunSpecifiedBenchmarks(nullptr, &reporter);

  // Shutdown Google Benchmark
  benchmark::Shutdown();
}



#endif