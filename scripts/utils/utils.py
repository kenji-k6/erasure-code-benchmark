from enum import Enum

Z_VALUE = 3.291 # 99.9% confidence interval

class Column:
  NAME              = "name"
  CPU_THREADS       = "cpu_threads"
  IS_GPU_COMPUTE    = "gpu_computation"
  GPU_BLOCKS        = "gpu_blocks"
  THREADS_PER_BLOCK = "threads_per_block"
  BLOCK_SIZE        = "block_size_KiB"
  EC                = "EC"
  LOST_BLOCKS       = "lost_blocks"
  ENC_THROUGHPUT    = "encode_throughput_Gbps" # Gbps
  DEC_THROUGHPUT    = "decode_throughput_Gbps" # Gbps
  ENC_THROUGHPUT_CI = "encode_throughput_Gbps_ci"
  DEC_THROUGHPUT_CI = "decode_throughput_Gbps_ci"

class Category(Enum):
  OPEN_SOURCE = "OPEN_SOURCE"
  SIMD = "SIMD"
  XOREC = "XOREC"
  HEATMAP = "HEATMAP"
  MULTI_THREAD = "MULTITHREAD"
  PAPER_COMPARISON = "PAPER_COMPARISON"


class PlotType(Enum):
  ENCODE = "encode"
  DECODE = "decode"


ALGORITHMS = "algorithms"
OUTPUT_DIR = "output_dir"
FILE_PREFIX = "file_prefix"
FIXED_VALS = "fixed_vals"

CATEGORY_INFO = {
  Category.OPEN_SOURCE: {
    ALGORITHMS: [
      "Leopard",
      "ISA-L",
      "CM256"
    ],
    OUTPUT_DIR: "opensource",
    FILE_PREFIX: "os",
    FIXED_VALS: {
      Column.BLOCK_SIZE: 4,
      Column.LOST_BLOCKS: 0,
      Column.EC: "(36/32)",
      Column.CPU_THREADS: 1,
    }
  },
  Category.SIMD: {
    ALGORITHMS: [
      "XOR-EC, Scalar",
      "XOR-EC, SSE2",
      "XOR-EC, AVX2",
      "XOR-EC, AVX-512"
    ],
    OUTPUT_DIR: "simd",
    FILE_PREFIX: "simd",
    FIXED_VALS: {
      Column.BLOCK_SIZE: 4,
      Column.LOST_BLOCKS: 0,
      Column.EC: "(36/32)",
      Column.CPU_THREADS: 1,
    }
  },
  Category.XOREC: {
    ALGORITHMS: [
      "XOR-EC, AVX2",
      "XOR-EC (Unified Memory), AVX2",
      "XOR-EC (GPU Memory), AVX2",
      "XOR-EC (GPU Computation)"
    ],
    OUTPUT_DIR: "xorec",
    FILE_PREFIX: "xorec",
    FIXED_VALS: {
      Column.BLOCK_SIZE: 4,
      Column.LOST_BLOCKS: 0,
      Column.EC: "(36/32)",
      Column.GPU_BLOCKS: 256,
      Column.THREADS_PER_BLOCK: 512,
      Column.CPU_THREADS: 1,
    }
  },
  Category.HEATMAP: {
    ALGORITHMS: [
      "XOR-EC, AVX2",
      "ISA-L",
      "CM256",
      "Leopard"
    ],
    OUTPUT_DIR: "misc",
    FILE_PREFIX: "heatmap",
    FIXED_VALS: {
      Column.CPU_THREADS: 1,
    }
  },
  Category.MULTI_THREAD: {
    ALGORITHMS: [
      "XOR-EC, AVX2",
      "ISA-L",
      "CM256",
      "Leopard"
    ],
    OUTPUT_DIR: "multithread",
    FILE_PREFIX: "multi",
    FIXED_VALS: {
      Column.BLOCK_SIZE: 4,
      Column.LOST_BLOCKS: 0,
      Column.EC: "(36/32)",
    }
  },
  Category.PAPER_COMPARISON: {
    ALGORITHMS: [
      "XOR-EC, AVX-512",
      "ISA-L"
    ],
    OUTPUT_DIR: "misc",
    FILE_PREFIX: "cmp",
    FIXED_VALS: {
      Column.BLOCK_SIZE: 64,
      Column.LOST_BLOCKS: 0,
      Column.EC: "(40/32)"
    }
  },
}




