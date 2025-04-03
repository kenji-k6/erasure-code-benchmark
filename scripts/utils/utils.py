import os
from cycler import V
import pandas as pd
from enum import Enum

import utils.config as cfg
from utils.hardware_info import CPUInfo

class AxType(Enum):
  ENCODE_T = 0
  ENCODE_TP = 2
  BLK_SIZE = 3
  FEC_RATIO = 4
  GPU_PARAMS = 5
  CPU_THREADS = 6

FILE_NAME_MAP = {
  AxType.ENCODE_T: "t",
  AxType.ENCODE_TP: "tp",
  AxType.BLK_SIZE: "blksize",
  AxType.FEC_RATIO: "fec",
  AxType.GPU_PARAMS: "gpu",
  AxType.CPU_THREADS: "threads"
}

COL_NAME_MAP = {
  AxType.ENCODE_T: "time_ms",
  AxType.ENCODE_TP: "throughput_Gbps",
  AxType.BLK_SIZE: "block_size_KiB",
  AxType.FEC_RATIO: "FEC",
  AxType.GPU_PARAMS: "gpu_params",
  AxType.CPU_THREADS: "num_cpu_threads"
}

LABEL_MAP = {
  AxType.ENCODE_T: "Encode Time (ms)",
  AxType.ENCODE_TP: "Encode Throughput (Gbps)",
  AxType.BLK_SIZE: "Block Size (KiB)",
  AxType.FEC_RATIO: "FEC Ratio",
  AxType.GPU_PARAMS: "(GPU Blocks, Threads/Block)",
  AxType.CPU_THREADS: "Num. CPU Threads"
}

AX_ARG_MAP = {
  "encode-time": AxType.ENCODE_T,
  "encode-throughput": AxType.ENCODE_TP,
  "block-size": AxType.BLK_SIZE,
  "fec": AxType.FEC_RATIO,
  "gpu-params": AxType.GPU_PARAMS,
  "cpu-threads": AxType.CPU_THREADS
}

#### TODO: Below


def ensure_paths() -> None:
  """Ensure that the input/output directories exist."""
  for dir in [cfg.RAW_DIR, cfg.OUTPUT_DIR]:
    os.makedirs(dir, exist_ok=True)
  
  os.path.isfile(cfg.INPUT_FILE)
  return


def get_output_path(x_axis: AxType, y_axis: AxType, plot_gpu: bool, overwrite: bool) -> str:
  """Generate the file name & path for the plot image."""
  base_name = f"{FILE_NAME_MAP[x_axis]}_{FILE_NAME_MAP[y_axis]}" + ("_gpu" if plot_gpu else "")
  file_name = f"{base_name}.png"
  file_path = os.path.join(cfg.OUTPUT_DIR, file_name)

  if overwrite: return file_path

  counter = 1
  while os.path.exists(file_path):
    file_name = f"{base_name}_{counter}.png"
    file_path = os.path.join(cfg.OUTPUT_DIR, file_name)
    counter += 1

  return file_path

def get_col_name(ax: AxType) -> str:
  """Returns the column name for the given AxType."""
  return COL_NAME_MAP[ax]

def get_ax_label(ax: AxType) -> str:
  """Returns the axis label for the given AxType."""
  return LABEL_MAP[ax]

def get_axis(axis: str) -> AxType:
  """Extract the AxTypes from the input list."""
  return AX_ARG_MAP[axis]

def get_plot_title(df: pd.DataFrame, x_axis: AxType, cpu_info: CPUInfo) -> str:
  """Generate a  plot title containing all the constant parameters."""
  # Sanity check
  cpu_title = f"CPU: {cpu_info.model_name}"

  first_row = df.iloc[0]
  def get_size_str(size: int):
    if size < 1024:
      return f"{size} B"
    elif 1024 <= size < 1024**2:
      return f"{size // 1024} KiB"
    else:
      return f"{size // (1024**2)} MiB"
    
  base_title = f"Message size: {get_size_str(first_row['message_size_B'])}, Num. Iterations: {first_row['num_iterations']}"

  if x_axis == AxType.BLK_SIZE:
    ax_title = f"{first_row['num_cpu_threads']} {('Thread' if first_row['num_cpu_threads'] == 1 else 'Threads')}, {first_row['FEC']}"
  elif x_axis == AxType.FEC_RATIO:
    ax_title = f"{first_row['num_cpu_threads']} {('Thread' if first_row['num_cpu_threads'] == 1 else 'Threads')}, Block Size: {get_size_str(first_row['block_size_B'])}"
  elif x_axis == AxType.CPU_THREADS:
    ax_title = f"Block Size: {get_size_str(first_row['block_size_B'])}, {first_row['FEC']}"
  else:
    raise ValueError(f"Invalid x_axis: {x_axis}")
  
  return f"{base_title}, {ax_title},\n{cpu_title}"