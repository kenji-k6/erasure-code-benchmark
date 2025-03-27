import os
import pandas as pd
from enum import Enum, IntEnum
from typing import Union, Tuple

import utils.config as cfg
from utils.hardware_info import CPUInfo

class AxType(Enum):
  ENCODE_T = 0
  ENCODE_TP = 2
  BLK_SIZE = 3
  FEC_RATIO = 4
  GPU_PARAMS = 5

FILE_NAME_MAP = {
  AxType.ENCODE_T: "encodetime",
  AxType.ENCODE_TP: "encodethroughput",
  AxType.BLK_SIZE: "buffersize",
  AxType.FEC_RATIO: "fecratio",
  AxType.GPU_PARAMS: "gpuparams"
}

COL_NAME_MAP = {
  AxType.ENCODE_T: "time_ms",
  AxType.ENCODE_TP: "throughput_Gbps",
  AxType.BLK_SIZE: "block_size_KiB",
  AxType.FEC_RATIO: "FEC",
  AxType.GPU_PARAMS: "gpu_params"
}

LABEL_MAP = {
  AxType.ENCODE_T: "Encode Time (ms)",
  AxType.ENCODE_TP: "Encode Throughput (Gbps)",
  AxType.BLK_SIZE: "Block Size (KiB)",
  AxType.FEC_RATIO: "FEC Ratio",
  AxType.GPU_PARAMS: "(GPU Blocks, Threads/Block)"
}

AX_ARG_MAP = {
  "encode-time": AxType.ENCODE_T,
  "encode-throughput": AxType.ENCODE_TP,
  "block-size": AxType.BLK_SIZE,
  "fec": AxType.FEC_RATIO,
  "gpu-params": AxType.GPU_PARAMS
}

#### TODO: Below


def ensure_paths() -> None:
  """Ensure that the input/output directories exist."""
  for dir in [cfg.RAW_DIR, cfg.OUTPUT_DIR]:
    os.makedirs(dir, exist_ok=True)
  
  os.path.isfile(cfg.INPUT_FILE)
  return


def get_output_path(x_axis: AxType, y_axis: AxType) -> str:
  """Generate the file name & path for the plot image."""

  file_name = f"{FILE_NAME_MAP[x_axis]}_vs_{FILE_NAME_MAP[y_axis]}.png"
  return os.path.join(cfg.OUTPUT_DIR, file_name)


def get_col_name(ax: AxType) -> str:
  """Returns the column name for the given AxType."""
  return COL_NAME_MAP[ax]


def get_ax_label(ax: AxType) -> str:
  """Returns the axis label for the given AxType."""
  return LABEL_MAP[ax]




def get_axis(axis: str) -> AxType:
  """Extract the AxTypes from the input list."""
  return AX_ARG_MAP[axis]


def get_fixed_param(x_axis: AxType) -> Tuple[str, Union[int, str]]:
  """Returns the fixed parameter for the given AxType."""
  if x_axis == AxType.BLK_SIZE:
    return get_col_name(AxType.FEC_RATIO), cfg.FIXED_FEC_RATIO
  elif x_axis == AxType.FEC_RATIO:
    return get_col_name(AxType.BLK_SIZE), cfg.FIXED_BLOCK_SIZE


def get_plot_title(df: pd.DataFrame, x_axis: AxType, cpu_info: CPUInfo) -> str:
  """Generate a  plot title containing all the constant parameters."""
  # Sanity check
  fixed_col, _ =  get_fixed_param(x_axis)
  assert(len(df[fixed_col].unique()) == 1)
  cpu_title = f"CPU: {cpu_info.model_name}"

  first_row = df.iloc[0]
  def get_size_str(size: int):
    if size < 1024:
      return f"{size} B"
    elif 1024 <= size < 1024**2:
      return f"{size // 1024} KiB"
    else:
      return f"{size // (1024**2)} MiB"

  titles = {
    AxType.BLK_SIZE: f"Message size: {get_size_str(first_row['message_size_B'])}, {first_row['FEC']}, Num. Iterations: {first_row['num_iterations']}",
    AxType.FEC_RATIO: f"Message size: {get_size_str(first_row['message_size_B'])}, Block Size: {get_size_str(first_row['block_size_B'])}, Num. Iterations: {first_row['num_iterations']}",
  }

  return f"{titles[x_axis]}\n{cpu_title}"