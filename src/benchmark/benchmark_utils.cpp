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
bool CPU_MEM = true;
bool GPU_MEM = false;
bool MEM_WARM = false;
bool MEM_COLD = true;

std::unordered_set<std::string> selected_benchmarks;
const std::unordered_map<std::string, BenchmarkFunction> available_benchmarks = {
  { "xor-ec",         BM_XOREC        },
  { "xor-ec-scalar",  BM_XOREC_Scalar },
  { "xor-ec-avx",     BM_XOREC_AVX    },
  { "xor-ec-avx2",    BM_XOREC_AVX2   },
  { "cm256",          BM_CM256        },
  { "isa-l",          BM_ISAL         },
  { "leopard",        BM_Leopard      },
  { "wirehair",       BM_Wirehair     }
};

const std::unordered_map<std::string, BenchmarkFunction> available_gpu_benchmarks = {
  { "xor-ec",         BM_XOREC_GPU        },
  { "xor-ec-scalar",  BM_XOREC_Scalar_GPU },
  { "xor-ec-avx",     BM_XOREC_AVX_GPU    },
  { "xor-ec-avx2",    BM_XOREC_AVX2_GPU   }
};

const std::unordered_map<std::string, std::string> benchmark_names = {
  { "xor-ec",         "XOR-EC (Auto)"   },
  { "xor-ec-scalar",  "XOR-EC (Scalar)" },
  { "xor-ec-avx",     "XOR-EC (AVX)"    },
  { "xor-ec-avx2",    "XOR-EC (AVX2)"   },
  { "cm256",          "CM256"           },
  { "isa-l",          "ISA-L"           },
  { "leopard",        "Leopard"         },
  { "wirehair",       "Wirehair"        }
};

static void usage() {
  std::cerr << "Usage: ec-benchmark [options]\n\n"

            << " Help Option:\n"
            << "  -h, --help                          show this help message\n\n"
            
            << " Benchmark Options:\n"
            << "  -i, --iterations=<num>              number of benchmark iterations (default 10)\n"
            << "      --full                          run the full benchmark suite, write results to CSV,\n"
            << "                                      any specifed benchmark config parameters will be ignored\n\n"
      
            << " Algorithm Selection:\n"
            << "      --xor-ec                        run the XOR-EC benchmark (automatically chooses between\n"
            << "                                      the Scalar, AVX, and AVX2 implementations according\n"
            << "                                      to the system specification)\n"
            << "      --xor-ec-scalar                 run the scalar XOR-EC implementation\n"
            << "                                      (SIMD optimizations disabled)\n"
            << "      --xor-ec-avx                    run the AVX XOR-EC implementation\n"
            << "      --xor-ec-avx2                   run the AVX2 XOR-EC implementation\n"
            << "      --cm256                         run the CM256 benchmark\n"
            << "      --isa-l                         run the ISA-L benchmark\n"
            << "      --leopard                       run the Leopard benchmark\n"
            << "      --wirehair                      run the Wirehair benchmark\n"
            << " *If no algorithm is specified, all algorithms will be run.*\n\n"

            << " Full Suite Options:\n"
            << "      --file=<file_name>              specify output file name\n"
            << "      --append                        append results to the output file (default: overwrite)\n"
            << "      --memory <gpu|cpu|all>          allocate the data buffers of the specified algorithms\n"
            << "                                      in GPU or CPU memory, or both (default: CPU)\n"
            << "      --memory-state <warm|cold|all>  whether to simulate cold CPU memory or warm CPU memory\n"
            << "                                      for encoding and decoding benchmarks (default: cold)\n"
            << "                                      *only relevant if --memory=gpu or --memory=all*\n"

            << " Single Run Options:\n"
            << "  -s, --size=<size>                   total size of original data in bytes\n"
            << "                                      default: " << FIXED_BUFFFER_SIZE << " B\n"
            << "  -b, --block-size=<size>             size of each block in bytes\n"
            << "                                      default: " << FIXED_BUFFFER_SIZE / FIXED_NUM_ORIGINAL_BLOCKS << " B\n"
            << "  -l, --lost_blocks=<num>             number of lost blocks\n"
            << "                                      default: " << FIXED_NUM_LOST_BLOCKS << '\n'
            << "  -r, --redundancy=<ratio>            redundancy ratio (#recovery blocks / #original blocks)\n"
            << "                                      default= " << FIXED_PARITY_RATIO << "\n\n";
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

  // XOR-EC checks
  if (selected_benchmarks.contains("xor-ec") ||
      selected_benchmarks.contains("xor-ec-scalar") ||
      selected_benchmarks.contains("xor-ec-avx") ||
      selected_benchmarks.contains("xor-ec-avx2")) {
    if (b % ECLimits::XOREC_BLOCK_ALIGNMENT != 0) {
      std::cerr << "Error: Block size must be a multiple of " << ECLimits::XOREC_BLOCK_ALIGNMENT << " for XOR-EC.\n";
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
  if (selected_benchmarks.contains("isa-l")) {
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

    if (CPU_MEM) {
      BenchmarkConfig cpu_config = config;
      cpu_config.gpu_mem = false;
      configs.push_back(cpu_config);
    }

    if (GPU_MEM) {
      BenchmarkConfig gpu_config = config;
      gpu_config.gpu_mem = true;

      if (MEM_WARM) {
        BenchmarkConfig warm_config = gpu_config;
        warm_config.mem_cold = false;
        configs.push_back(warm_config);
      }
      if (MEM_COLD) {
        BenchmarkConfig cold_config = gpu_config;
        cold_config.mem_cold = true;
        configs.push_back(cold_config);
      }
    }
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

    if (CPU_MEM) {
      BenchmarkConfig cpu_config = config;
      cpu_config.gpu_mem = false;
      configs.push_back(cpu_config);
    }

    if (GPU_MEM) {
      BenchmarkConfig gpu_config = config;
      gpu_config.gpu_mem = true;

      if (MEM_WARM) {
        BenchmarkConfig warm_config = gpu_config;
        warm_config.mem_cold = false;
        configs.push_back(warm_config);
      }
      if (MEM_COLD) {
        BenchmarkConfig cold_config = gpu_config;
        cold_config.mem_cold = true;
        configs.push_back(cold_config);
      }
    }
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

    if (CPU_MEM) {
      BenchmarkConfig cpu_config = config;
      cpu_config.gpu_mem = false;
      configs.push_back(cpu_config);
    }

    if (GPU_MEM) {
      BenchmarkConfig gpu_config = config;
      gpu_config.gpu_mem = true;

      if (MEM_WARM) {
        BenchmarkConfig warm_config = gpu_config;
        warm_config.mem_cold = false;
        configs.push_back(warm_config);
      }
      if (MEM_COLD) {
        BenchmarkConfig cold_config = gpu_config;
        cold_config.mem_cold = true;
        configs.push_back(cold_config);
      }
    }
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
  config.gpu_mem = false;
  config.mem_cold = false;
  return config;
}

static void inline add_benchmark(std::string name) {
  if (name == "xor-ec-avx") {
    #if defined(__AVX__)
      selected_benchmarks.insert(name);
    #else
      std::cerr << "Error: AVX is not supported on this system.\n";
      exit(0);
    #endif
  }

  if (name == "xor-ec-avx2") {
    #if defined(__AVX2__)
      selected_benchmarks.insert(name);
    #else
      std::cerr << "Error: AVX2 is not supported on this system.\n";
      exit(0);
    #endif
  }
  selected_benchmarks.insert(name);
}

void get_configs(int argc, char** argv, std::vector<BenchmarkConfig>& configs, std::vector<uint32_t>& lost_block_idxs) {
  struct option long_options[] = {
    { "help",             no_argument,        nullptr, 'h'  },
    { "iterations",       required_argument,  nullptr, 'i'  },
    { "full",             no_argument,        nullptr,  0   },
    { "memory",           required_argument,  nullptr,  0   },
    { "memory-state",     required_argument,  nullptr,  0   },
    { "file",             required_argument,  nullptr,  0   },
    { "append",           no_argument,        nullptr,  0   },
    { "xor-ec",           no_argument,        nullptr,  0   },
    { "xor-ec-scalar",    no_argument,        nullptr,  0   },
    { "xor-ec-avx",       no_argument,        nullptr,  0   },
    { "xor-ec-avx2",      no_argument,        nullptr,  0   },
    { "cm256",            no_argument,        nullptr,  0   },
    { "isa-l",            no_argument,        nullptr,  0   },
    { "leopard",          no_argument,        nullptr,  0   },
    { "wirehair",         no_argument,        nullptr,  0   },
    { "size",             no_argument,        nullptr, 's'  },
    { "block-size",       no_argument,        nullptr, 'b'  },
    { "lost-blocks",      no_argument,        nullptr, 'l'  },
    { "redundancy",       no_argument,        nullptr, 'r'  },
    { nullptr,            0,                  nullptr,  0   }
  };

  uint64_t s = FIXED_BUFFFER_SIZE;
  uint64_t b = FIXED_BUFFFER_SIZE / FIXED_NUM_ORIGINAL_BLOCKS;
  uint32_t l = FIXED_NUM_LOST_BLOCKS;
  double r = FIXED_PARITY_RATIO;
  int i = 10;

  int c;
  int option_index = 0;
  std::string name;
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
        name = std::string(long_options[option_index].name);

        if (name == "full") {
          FULL_BENCHMARK = true;

        } else if (name == "memory") {
          auto arg = std::string(optarg);
          if (arg == "gpu") {
            CPU_MEM = false;
            GPU_MEM = true;
          } else if (arg == "cpu") {
            CPU_MEM = true;
            GPU_MEM = false;
          } else if (arg == "all") {
            CPU_MEM = true;
            GPU_MEM = true;
          } else {
            std::cerr << "Error: --memory option must be either 'gpu', 'cpu', or 'all'.\n";
            exit(0);
          }

        } else if (name == "memory-state") {
          auto arg = std::string(optarg);
          if (arg == "warm") {
            MEM_WARM = true;
            MEM_COLD = false;
          } else if (arg == "cold") {
            MEM_WARM = false;
            MEM_COLD = true;
          } else if (arg == "all") {
            MEM_WARM = true;
            MEM_COLD = true;
          } else {
            std::cerr << "Error: --memory-state option must be either 'warm', 'cold', or 'all'.\n";
            exit(0);
          }

        } else if (name == "file") {
          output_file_name = std::string(optarg);

        } else if (name == "append") {
          OVERWRITE_FILE = false;

        } else {
          add_benchmark(name);
        }
        break;
      default:
        usage();
        exit(0);
    }
  }

  if (selected_benchmarks.empty()) {
    for (const auto& [name, _] : available_benchmarks) {
      if (name != "xor-ec") add_benchmark(name);
    }
  }

  if (FULL_BENCHMARK) {
    get_full_benchmark_configs(i, configs, lost_block_idxs);
  } else {
    configs.push_back(get_single_benchmark_config(s, b, l, r, i, lost_block_idxs));
  }
}

static std::string get_benchmark_name(std::string inp, bool gpu_mem, bool mem_cold) {
  auto base_name = benchmark_names.at(inp);
  if (gpu_mem) {
    base_name += "(GPU, ";
    if (mem_cold) {
      base_name += "Cold)";
    } else {
      base_name += "Warm)";
    }
  }
  return base_name;
}

static BenchmarkFunction get_benchmark_func(std::string inp_name, bool gpu_mem) {
  if (gpu_mem) {
    return available_gpu_benchmarks.at(inp_name);
  }
  return available_benchmarks.at(inp_name);
}

static void register_benchmarks(std::vector<BenchmarkConfig>& configs, BenchmarkProgressReporter *console_reporter) {
  for (auto& config: configs) {
    config.progress_reporter = console_reporter;

    for (auto& inp_name : selected_benchmarks) {
      auto bm_name = get_benchmark_name(inp_name, config.gpu_mem, config.mem_cold);
      auto bm_func = get_benchmark_func(inp_name, config.gpu_mem);

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