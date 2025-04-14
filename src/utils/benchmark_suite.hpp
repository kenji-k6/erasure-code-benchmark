/**
 * @file benchmark_suite.hpp
 * @brief Utility functions and constants for parsing and validating command-line arguments
 */

#ifndef BM_SUITE_HPP
#define BM_SUITE_HPP
#include "bm_config.hpp"
#include "runners.hpp"
#include <unordered_map>

using BenchmarkTuple = std::tuple<std::string, BenchmarkFunction, BenchmarkConfig>;

const std::unordered_map<std::string, BenchmarkFunction> CPU_BM_FUNCTIONS = {
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
  { "xorec",      BM_XOREC_GPU_CMP }
};

const std::unordered_map<std::string, std::string> GPU_BM_NAMES = {
  { "xorec",  "XOR-EC (GPU Computation)"  }
};

const std::unordered_map<std::string, XorecVersion> XOREC_VERSIONS = {
  { "scalar",  XorecVersion::Scalar  },
  { "avx",     XorecVersion::AVX     },
  { "avx2",    XorecVersion::AVX2    },
  { "avx512",  XorecVersion::AVX512  }
};



void run_benchmarks(int argc, char** argv);
#endif // BM_SUITE_HPP