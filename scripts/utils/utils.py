from enum import Enum
Z_VALUE = 3.291 # 99.9% confidence interval

class Column:
  NAME              = "name"
  IS_GPU_COMPUTE    = "gpu_computation"
  GPU_BLOCKS        = "gpu_blocks"
  THREADS_PER_BLOCK = "threads_per_block"
  DATA_SIZE         = "data_size_KiB"  # B
  EC                = "EC"
  LOST_BLOCKS       = "lost_blocks"
  ENC_THROUGHPUT    = "encode_throughput_Gbps" # Gbps
  DEC_THROUGHPUT    = "decode_throughput_Gbps" # Gbps
  ENC_THROUGHPUT_CI = "encode_throughput_Gbps_ci"
  DEC_THROUGHPUT_CI = "decode_throughput_Gbps_ci"

class Category(Enum):
  CPU = "CPU"
  OPENSOURCE = "OPEN_SOURCE"
  SIMD = "XOREC SIMD"
  XOREC = "XOREC VARIANT"

class PlotType(Enum):
  ENCODE = "encode"
  DECODE = "decode"

OUTPUT_SUBDIR = {
  Category.CPU: "cpu",
  Category.OPENSOURCE: "opensource",
  Category.SIMD: "simd",
  Category.XOREC: "xorec"
}

CATEGORY_INFO = {
  Category.CPU: {
    "algorithms": [
      "XOR-EC, AVX2",
      "ISA-L",
      "CM256",
      "Leopard"
    ],
    "file_prefix": "cpu",
    "fixed_vals": {
      Column.DATA_SIZE: 512,
      Column.LOST_BLOCKS: 0,
      Column.EC: "(36/32)",
    }
  },
  Category.OPENSOURCE: {
    "algorithms": [
      "ISA-L",
      "CM256",
      "Leopard"
    ],
    "file_prefix": "os",
    "fixed_vals": {
      Column.DATA_SIZE: 512,
      Column.LOST_BLOCKS: 0,
      Column.EC: "(36/32)",
    }
  },
  Category.SIMD: {
    "algorithms": [
      "XOR-EC, Scalar",
      "XOR-EC, SSE2",
      "XOR-EC, AVX2",
      "XOR-EC, AVX-512"
    ],
    "file_prefix": "simd",
    "fixed_vals": {
      Column.DATA_SIZE: 512,
      Column.LOST_BLOCKS: 0,
      Column.EC: "(36/32)",
    }
  },
  Category.XOREC: {
    "algorithms": [
      "XOR-EC, AVX2",
      "XOR-EC (Unified Memory), AVX2",
      "XOR-EC (GPU Memory), AVX2",
      "XOR-EC (GPU Computation)"
    ],
    "file_prefix": "xorec",
    "fixed_vals": {
      Column.DATA_SIZE: 512,
      Column.LOST_BLOCKS: 0,
      Column.EC: "(36/32)",
      Column.GPU_BLOCKS: 256,
      Column.THREADS_PER_BLOCK: 512
    }
  }
}




