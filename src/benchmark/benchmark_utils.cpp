/**
 * @file benchmark_utils.cpp
 * @brief Implementations of utility functions for parsing and validating command-line arguments
 */

#include "benchmark/benchmark.h"
#include "benchmark_utils.h"
#include "benchmark_functions.h"
#include "utils.h"
#include <cmath>
#include <iostream>
#include <getopt.h>

// Global variable definitions
constexpr const char* OUTPUT_FILE_DIR = "../results/raw/";
std::string output_file_name = "benchmark_results.csv";
bool FULL_BENCHMARK = false;
bool OVERWRITE_FILE = true;

std::unordered_set<std::string> selected_benchmarks;
const std::unordered_map<std::string, BenchmarkFunction> available_benchmarks = {
  { "baseline",         BM_Baseline       },
  { "baseline-scalar",  BM_BaselineScalar },
  { "baseline-avx",     BM_BaselineAVX    },
  { "baseline-avx2",    BM_BaselineAVX2   },
  { "cm256",            BM_CM256          },
  { "isal",             BM_ISAL           },
  { "leopard",          BM_Leopard        },
  { "wirehair",         BM_Wirehair       }
};

const std::unordered_map<std::string, std::string> benchmark_names = {
  { "baseline",         "Baseline (Auto)"   },
  { "baseline-scalar",  "Baseline (Scalar)" },
  { "baseline-avx",     "Baseline (AVX)"    },
  { "baseline-avx2",    "Baseline (AVX2)"   },
  { "cm256",            "CM256"             },
  { "isal",             "ISA-L"             },
  { "leopard",          "Leopard"           },
  { "wirehair",         "Wirehair"          }
};

static void usage() {
 std::cerr << "Usage: ec-benchmark [options]\n"
           << "  -h | --help       Help\n"
           << "  -i <num>          Number of benchmark iterations (default 10)\n\n"

           << "  --full            Run the full benchmark config suite, any specified benchmark config\n"
           << "                    parameters will be ignored\n"
           << "  --file <name>     Name of the output file (default: " << output_file_name << ")\n"
           << "                    Only used when --full is specified\n"
           << "  --append          If set, the outputs will be appended to the file, if not set it will\n"
           << "                    overwrite. Only used when --full is specified\n\n"

           << " The following flags are used to specify which benchmarks to run.\n"
           << "  --baseline        Run the baseline benchmark (automatically chooses between the Scalar,\n"
           << "                    AVX, and AVX2 implementations according to system specification).\n"
           << "  --baseline-scalar Run the scalar baseline implementation\n"
           << "  --baseline-avx    Run the AVX baseline implementation\n"
           << "  --baseline-avx2   Run the AVX2 baseline implementation\n"
           << "  --cm256           Run the CM256 benchmark\n"
           << "  --isal            Run the ISA-L benchmark\n"
           << "  --leopard         Run the Leopard benchmark\n"
           << "  --wirehair        Run the Wirehair benchmark\n\n"

           << " The following flags are used to specify the parameters if --full was not specified.\n"
           << " *Nothing will be written to file.*\n"
           << "  -s <size>         Total size of original data in bytes (default: " << FIXED_BUFFFER_SIZE << "B)\n"
           << "  -b <size>         Size of each block in bytes (default: " << FIXED_BUFFFER_SIZE / FIXED_NUM_ORIGINAL_BLOCKS << "B)\n" 
           << "  -l <num>          Number of lost blocks (default: " << FIXED_NUM_LOST_BLOCKS << ")\n"
           << "  -r <ratio>        Redundancy ratio (#recovery blocks / #original blocks) (default: " << FIXED_PARITY_RATIO << "\n\n"


           << " *If no arguments at all are specified, the full selection of benchmarks, over multiple configurations, will be run.*\n";
 exit(0);
}

static void check_args(uint64_t s, uint64_t b, uint32_t l, double r, int i, uint32_t num_orig_blocks, uint32_t num_rec_blocks) {
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
    if (b % ECLimits::BASELINE_BLOCK_ALIGNMENT != 0) {
      std::cerr << "Error: Block size must be a multiple of " << ECLimits::BASELINE_BLOCK_ALIGNMENT << " for Baseline.\n";
      exit(0);
    }
  }
  
  // CM256 Checks
  if (selected_benchmarks.contains("cm256")) {
    if (num_orig_blocks + num_rec_blocks > ECLimits::CM256_MAX_TOT_BLOCKS) {
      std::cerr << "Error: Total number of blocks exceeds the maximum allowed by CM256.\n"
                << "Condition: #(original blocks) + #(recovery blocks) <= " << ECLimits::CM256_MAX_TOT_BLOCKS << "\n"
                << "(#original blocks) = " << num_orig_blocks << ", (#recovery blocks) = " << num_rec_blocks << "\n";
      exit(0);
    }
  }

  // ISA-L Checks
  if (selected_benchmarks.contains("isal")) {
    if (num_orig_blocks + num_rec_blocks > ECLimits::ISAL_MAX_TOT_BLOCKS) {
      std::cerr << "Error: Total number of blocks exceeds the maximum allowed by ISA-L.\n"
                << "Condition: #(original blocks) + #(recovery blocks) <= " << ECLimits::ISAL_MAX_TOT_BLOCKS << "\n"
                << "(#original blocks) = " << num_orig_blocks << ", (#recovery blocks) = " << num_rec_blocks << "\n";
      exit(0);
    }
  }


  // Leopard Checks
  if (selected_benchmarks.contains("leopard")) {
    if (num_orig_blocks + num_rec_blocks > ECLimits::LEOPARD_MAX_TOT_BLOCKS) {
      std::cerr << "Error: Total number of blocks exceeds the maximum allowed by Leopard.\n"
                << "Condition: #(original blocks) + #(recovery blocks) <= " << ECLimits::LEOPARD_MAX_TOT_BLOCKS << "\n"
                << "(#original blocks) = " << num_orig_blocks << ", (#recovery blocks) = " << num_rec_blocks << "\n";
      exit(0);
    }

    if (num_rec_blocks > num_orig_blocks) {
      std::cerr << "Error: Number of recovery blocks can't exceed the number of original blocks for Leopard.\n"
                << "Condition: #(recovery blocks) <= #(original blocks)\n"
                << "(#recovery blocks) = " << num_rec_blocks << ", (#original blocks) = " << num_orig_blocks << "\n";
      exit(0);
    }

    if (b % ECLimits::LEOPARD_BLOCK_ALIGNMENT != 0) {
      std::cerr << "Error: Block size must be a multiple of " << ECLimits::LEOPARD_BLOCK_ALIGNMENT << " for Leopard.\n";
      exit(0);
    } 
  }

  // Wirehair Checks
  if (selected_benchmarks.contains("wirehair")) {
    if (num_orig_blocks < ECLimits::WIREHAIR_MIN_DATA_BLOCKS || num_orig_blocks > ECLimits::WIREHAIR_MAX_DATA_BLOCKS) {
      std::cerr << "Error: Number of original blocks must be between " << ECLimits::WIREHAIR_MIN_DATA_BLOCKS << " and " << ECLimits::WIREHAIR_MAX_DATA_BLOCKS << " for Wirehair.\n"
                << "(#original blocks) = " << num_orig_blocks << "\n";
      exit(0);
    }
  }
}

// Function that creates the configs for the full benchmark suite
static void get_full_benchmark_configs(int num_iterations, std::vector<BenchmarkConfig>& configs, std::vector<uint32_t>& lost_block_idxs) {
  int tot_lost_blocks = FIXED_NUM_LOST_BLOCKS;
  for (auto num_lost_blocks : VAR_NUM_LOST_BLOCKS) {
    tot_lost_blocks += num_lost_blocks;
  }
  lost_block_idxs.resize(tot_lost_blocks);

  // Select the lost block indices for the first two plots
  select_lost_block_idxs(
    FIXED_NUM_RECOVERY_BLOCKS,
    FIXED_NUM_LOST_BLOCKS,
    FIXED_NUM_ORIGINAL_BLOCKS + FIXED_NUM_RECOVERY_BLOCKS,
    lost_block_idxs.data()
  );

  // Configs for varying buffer sizes
  select_lost_block_idxs(
    FIXED_NUM_RECOVERY_BLOCKS,
    FIXED_NUM_LOST_BLOCKS,
    FIXED_NUM_ORIGINAL_BLOCKS + FIXED_NUM_RECOVERY_BLOCKS,
    lost_block_idxs.data()
  );

  for (auto buf_size : VAR_BUFFER_SIZE) {
    BenchmarkConfig config;
    config.data_size = buf_size;
    config.block_size = buf_size / FIXED_NUM_ORIGINAL_BLOCKS;
    config.num_lost_blocks = FIXED_NUM_LOST_BLOCKS;
    config.redundancy_ratio = FIXED_PARITY_RATIO;
    config.num_iterations = num_iterations;
    config.plot_id = 0;
    config.lost_block_idxs = lost_block_idxs.data();
    config.computed.num_original_blocks = FIXED_NUM_ORIGINAL_BLOCKS;
    config.computed.num_recovery_blocks = FIXED_NUM_RECOVERY_BLOCKS;
    
    configs.push_back(config);
  }

  // Configs for varying redundancy ratio
  for (auto num_rec_blocks : VAR_NUM_RECOVERY_BLOCKS) {
    BenchmarkConfig config;
    config.data_size = FIXED_BUFFFER_SIZE;
    config.block_size = FIXED_BUFFFER_SIZE / FIXED_NUM_ORIGINAL_BLOCKS;
    config.num_lost_blocks = FIXED_NUM_LOST_BLOCKS;
    config.redundancy_ratio = static_cast<double>(num_rec_blocks) / FIXED_NUM_ORIGINAL_BLOCKS;
    config.num_iterations = num_iterations;
    config.plot_id = 1;
    config.lost_block_idxs = lost_block_idxs.data();
    config.computed.num_original_blocks = FIXED_NUM_ORIGINAL_BLOCKS;
    config.computed.num_recovery_blocks = num_rec_blocks;

    configs.push_back(config);
  }

  // Configs for varying no. of lost blocks.
  uint32_t *curr = lost_block_idxs.data() + FIXED_NUM_LOST_BLOCKS;

  for (auto num_lost_blocks : VAR_NUM_LOST_BLOCKS) {
    select_lost_block_idxs(
      FIXED_NUM_ORIGINAL_BLOCKS,
      num_lost_blocks,
      FIXED_NUM_ORIGINAL_BLOCKS + num_lost_blocks,
      curr
    );

    BenchmarkConfig config;
    config.data_size = FIXED_BUFFFER_SIZE;
    config.block_size = FIXED_BUFFFER_SIZE / FIXED_NUM_ORIGINAL_BLOCKS;
    config.num_lost_blocks = num_lost_blocks;
    config.redundancy_ratio = 1.0;
    config.num_iterations = num_iterations;
    config.plot_id = 2;
    config.lost_block_idxs = curr;
    config.computed.num_original_blocks = FIXED_NUM_ORIGINAL_BLOCKS;
    config.computed.num_recovery_blocks = FIXED_NUM_ORIGINAL_BLOCKS;

    configs.push_back(config);
    curr += num_lost_blocks;
  }
}

static BenchmarkConfig get_single_benchmark_config(uint64_t s, uint64_t b, uint32_t l, double r, int i, std::vector<uint32_t>& lost_block_idxs) {
  uint32_t num_orig = (s + (b - 1)) / b;
  uint32_t num_rec = static_cast<size_t>(std::ceil(num_orig * r));
  check_args(s, b, l, r, i, num_orig, num_rec);
  lost_block_idxs.resize(l);
  select_lost_block_idxs(num_rec, l, num_orig + num_rec, lost_block_idxs.data());

  BenchmarkConfig config;
  config.data_size = s;
  config.block_size = b;
  config.num_lost_blocks = l;
  config.redundancy_ratio = r;
  config.num_iterations = i;
  config.plot_id = -1;
  config.lost_block_idxs = lost_block_idxs.data();
  config.computed.num_original_blocks = num_orig;
  config.computed.num_recovery_blocks = num_rec;

  return config;
}

void get_configs(int argc, char** argv, std::vector<BenchmarkConfig>& configs, std::vector<uint32_t>& lost_block_idxs) {
  struct option long_options[] = {
    { "help",             no_argument,        nullptr, 'h'  },
    { "full",             no_argument,        nullptr,  0   },
    { "file",             required_argument,  nullptr,  0   },
    { "append",           no_argument,        nullptr,  0   },
    { "baseline",         no_argument,        nullptr,  0   },
    { "baseline-scalar",  no_argument,        nullptr,  0   },
    { "baseline-avx",     no_argument,        nullptr,  0   },
    { "baseline-avx2",    no_argument,        nullptr,  0   },
    { "cm256",            no_argument,        nullptr,  0   },
    { "isal",             no_argument,        nullptr,  0   },
    { "leopard",          no_argument,        nullptr,  0   },
    { "wirehair",         no_argument,        nullptr,  0   },
    { nullptr,            0,                  nullptr,  0   }
  };

  uint64_t s = FIXED_BUFFFER_SIZE;
  uint64_t b = FIXED_BUFFFER_SIZE / FIXED_NUM_ORIGINAL_BLOCKS;
  uint32_t l = FIXED_NUM_LOST_BLOCKS;
  double r = FIXED_PARITY_RATIO;
  int i = 10;

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
        l = std::stoul(optarg);
        break;
      case 'r':
        r = std::stod(optarg);
        break;
      case 'i':
        i = std::stoi(optarg);
        break;
      case 0:
        if (long_options[option_index].name) {
          if (std::string(long_options[option_index].name) == "full") {
            FULL_BENCHMARK = true;
          } else if (std::string(long_options[option_index].name) == "file") {
            output_file_name = std::string(optarg);
          } else if (std::string(long_options[option_index].name) == "append") {
            OVERWRITE_FILE = false;
          }
          else {
            selected_benchmarks.insert(long_options[option_index].name);
          }
        }
        break;
      default:
        usage();
        exit(0);
    }
  }

  if (selected_benchmarks.empty()) {
    for (const auto& [name, func] : available_benchmarks) {
      if (name != "baseline") selected_benchmarks.insert(name);
    }
  }

  if (FULL_BENCHMARK) {
    get_full_benchmark_configs(i, configs, lost_block_idxs);
  } else {
    configs.push_back(get_single_benchmark_config(s, b, l, r, i, lost_block_idxs));
  }
}

static void register_benchmarks(std::vector<BenchmarkConfig>& configs, BenchmarkProgressReporter *console_reporter) {
  for (auto& config: configs) {
    config.progress_reporter = console_reporter;
    for (auto& inp_name : selected_benchmarks) {
      auto bm_name = benchmark_names.at(inp_name);
      auto bm_func = available_benchmarks.at(inp_name);
      benchmark::RegisterBenchmark(bm_name, bm_func, config)->UseManualTime()->Iterations(config.num_iterations);
    }
  }
}

void run_benchmarks(std::vector<BenchmarkConfig>& configs) {
  if (configs.empty()) exit(1);

  std::unique_ptr<BenchmarkProgressReporter> console_reporter;
  std::unique_ptr<BenchmarkCSVReporter> csv_reporter;

  int argc = 2;
  char *argv[] = { (char*)"benchmark", (char*)"--benchmark_out=console" };

  if (FULL_BENCHMARK) { // Full suite
    int tot_num_iterations = configs[0].num_iterations * configs.size() * selected_benchmarks.size();
    console_reporter = std::make_unique<BenchmarkProgressReporter>(tot_num_iterations);
    csv_reporter = std::make_unique<BenchmarkCSVReporter>(OUTPUT_FILE_DIR + output_file_name, OVERWRITE_FILE);
  } else { // Individual run
    argc = 1;
  }

  register_benchmarks(configs, console_reporter.get());

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return;
  benchmark::RunSpecifiedBenchmarks(console_reporter.get(), csv_reporter.get());
  benchmark::Shutdown();
}