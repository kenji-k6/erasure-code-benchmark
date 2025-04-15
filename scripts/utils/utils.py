Z_VALUE = 3.291 # 99.9% confidence interval

class Column:
  NAME              = "name"
  GPU_BLOCKS        = "gpu_blocks"
  THREADS_PER_BLOCK = "threads_per_block"
  DATA_SIZE         = "data_size_B"  # B
  BLOCK_SIZE        = "block_size_KiB"  # KiB
  EC                = "EC"
  LOST_BLOCKS       = "lost_blocks"
  ENC_THROUGHPUT    = "encode_throughput_Gbps" # Gbps
  DEC_THROUGHPUT    = "decode_throughput_Gbps" # Gbps
  ENC_THROUGHPUT_CI = "encode_throughput_Gbps_ci"
  DEC_THROUGHPUT_CI = "decode_throughput_Gbps_ci"


FIXED_VALS = {
  Column.BLOCK_SIZE: 128,
  Column.LOST_BLOCKS: 0,
}



