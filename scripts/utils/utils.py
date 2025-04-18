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
  GPU = "GPU"
  SIMD = "XOREC SIMD"
  XOREC = "XOREC VARIANT"

class PlotType(Enum):
  ENCODE = "encode"
  DECODE = "decode"

OUTPUT_SUBDIR = {
  Category.CPU: "cpu",
  Category.GPU: "gpu",
  Category.SIMD: "simd",
  Category.XOREC: "xorec"
}


FIXED_VALS = {
  Column.DATA_SIZE: 512,
  Column.LOST_BLOCKS: 0,
  Column.EC: "(36/32)"
}



CATEGORY_INFO = {
  Category.CPU: {
    "algorithms": [
      "Xorec, AVX2",
      "ISA-L",
      "CM256",
      "Leopard"
    ],
    "file_prefix": "cpu"
  },
  Category.GPU: {
    "algorithms": [
      "Xorec (GPU Computation)"
    ],
    "file_prefix": "xorec_gpu"
  },
  Category.SIMD: {
    "algorithms": [
      "Xorec, Scalar",
      "Xorec, AVX",
      "Xorec, AVX2",
      "Xorec, AVX512"
    ],
    "file_prefix": "simd"
  },
  Category.XOREC: {
    "algorithms": [
      "Xorec, AVX2",
      "Xorec (Unified Memory), AVX2",
      "Xorec (GPU Memory), AVX2",
    ],
    "file_prefix": "xorec"
  }
}




