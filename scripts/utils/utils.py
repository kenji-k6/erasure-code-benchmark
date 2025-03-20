import os
import pandas as pd
from enum import Enum, IntEnum
from typing import List

import utils.config as cfg
from utils.hardware_info import CPUInfo

class AxType(Enum):
  ENCODE_T = 0
  DECODE_T = 1
  ENCODE_TP = 2
  DECODE_TP = 3
  BUF_SIZE = 4
  PARITY_BLKS = 5
  LOST_BLKS = 6

class XorecVersion(IntEnum):
  Scalar = 0
  AVX = 1
  AVX2 = 2
  AVX512 = 3

FILE_NAME_MAP = {
  AxType.ENCODE_T: "encodetime",
  AxType.DECODE_T: "decodetime",
  AxType.ENCODE_TP: "encodethroughput",
  AxType.DECODE_TP: "decodethroughput",
  AxType.BUF_SIZE: "buffersize",
  AxType.PARITY_BLKS: "parityblocks",
  AxType.LOST_BLKS: "lostblocks"
}

COL_NAME_MAP = {
  AxType.ENCODE_T: "encode_time_ms",
  AxType.DECODE_T: "decode_time_ms",
  AxType.ENCODE_TP: "encode_throughput_Gbps",
  AxType.DECODE_TP: "decode_throughput_Gbps",
  AxType.BUF_SIZE: "tot_data_size_KiB",
  AxType.PARITY_BLKS: "num_parity_blocks",
  AxType.LOST_BLKS: "num_lost_blocks"
}

LABEL_MAP = {
  AxType.ENCODE_T: "Encode Time (ms)",
  AxType.DECODE_T: "Decode Time (ms)",
  AxType.ENCODE_TP: "Encode Throughput (Gbps)",
  AxType.DECODE_TP: "Decode Throughput (Gbps)",
  AxType.BUF_SIZE: "Buffer Size",
  AxType.PARITY_BLKS: "#Parity Blocks",
  AxType.LOST_BLKS: "#Lost Blocks"
}

PLOT_ID_MAP = {
  AxType.BUF_SIZE: 0,
  AxType.PARITY_BLKS: 1,
  AxType.LOST_BLKS: 2
}

BM_NAME_MAP = {
  "cm256": "CM256",
  "isal": "ISA-L",
  "leopard": "Leopard",
  "wirehair": "Wirehair",
  "xorec": "XOR-EC"
}

AX_ARG_MAP = {
  "buffer-size": AxType.BUF_SIZE,
  "num-parity-blocks": AxType.PARITY_BLKS,
  "num-lost-blocks": AxType.LOST_BLKS,
  "encode-time": AxType.ENCODE_T,
  "decode-time": AxType.DECODE_T,
  "encode-throughput": AxType.ENCODE_TP,
  "decode-throughput": AxType.DECODE_TP
}




def ensure_paths(check_perf_file: bool) -> None:
  """Ensure that the input/output directories exist."""
  for dir in [cfg.RAW_DIR, cfg.OUTPUT_DIR]:
    os.makedirs(dir, exist_ok=True)
  
  os.path.isfile(cfg.EC_INPUT_FILE)
  if check_perf_file:
    os.path.isfile(cfg.PERF_INPUT_FILE)
  return


def get_output_path(x_axis: AxType, y_axis: AxType, y_scale: str) -> str:
  """Generate the file name & path for the plot image."""
  if y_scale != "linear" and y_scale != "log":
    raise ValueError(f"Unsupported y_scale: {y_scale}")

  file_name = f"{FILE_NAME_MAP[x_axis]}_vs_{FILE_NAME_MAP[y_axis]}_{y_scale}.png"
  return os.path.join(cfg.OUTPUT_DIR, file_name)


def get_col_name(ax: AxType) -> str:
  """Returns the column name for the given AxType."""
  return COL_NAME_MAP[ax]


def get_ax_label(ax: AxType) -> str:
  """Returns the axis label for the given AxType."""
  return LABEL_MAP[ax]


def get_plot_title(df: pd.DataFrame, cpu_info: CPUInfo) -> str:
  """Generate a  plot title containing all the constant parameters."""
  first_row = df.iloc[0]
  plot_id = first_row["plot_id"]

  cpu_title = f"CPU: {cpu_info.model_name} @ {cpu_info.clock_rate_GHz} GHz"

  titles = {
    0: f"#Data Blocks: {first_row['num_data_blocks']}, #Redundancy Ratio: {first_row['redundancy_ratio']}, #Lost Blocks: {first_row['num_lost_blocks']}, #Iterations: {first_row['num_iterations']}",
    1: f"Buffer Size: {first_row['tot_data_size_KiB']} KiB, #Data Blocks: {first_row['num_data_blocks']}, #Lost Blocks: {first_row['num_lost_blocks']}, #Iterations: {first_row['num_iterations']}",
    2: f"Buffer Size: {first_row['tot_data_size_KiB']} KiB, #Data Blocks: {first_row['num_data_blocks']}, #Parity Blocks: {int(first_row['num_parity_blocks'])}, #Iterations: {first_row['num_iterations']}"
  }

  return f"{titles[plot_id]}\n{cpu_title}"


def get_plot_id(x_axis: AxType) -> int:
  """Returns the plot_id for the given x_axis."""
  return PLOT_ID_MAP[x_axis]


def get_benchmark_names(bms: List[str]) -> List[str]:
  """Extract the benchmark names from the input list."""
  return [BM_NAME_MAP[bm] for bm in bms]


def get_axes(axes: List[str]) -> List[AxType]:
  """Extract the AxTypes from the input list."""
  return [AX_ARG_MAP[ax] for ax in axes]