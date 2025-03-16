import os
import pandas as pd
from enum import Enum
from utils.config import RAW_DIR, OUTPUT_DIR
from utils.hardware_info import CPUInfo

class AxType(Enum):
  ENCODE_T = 0
  DECOCE_T = 1
  ENCODE_TP = 2
  DECODE_TP = 3
  BUF_SIZE = 4
  PARITY_BLKS = 5
  LOST_BLKS = 6


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


def ensure_dirs() -> None:
  """Ensure that the input/output directories exist."""
  for dir in [RAW_DIR, OUTPUT_DIR]:
    os.makedirs(dir, exist_ok=True)


def get_output_path(x_axis: AxType, y_axis: AxType, y_scale: str) -> str:
  """Generate the file name & path for the plot image."""
  if y_scale != "linear" and y_scale != "log":
    raise ValueError(f"Unsupported y_scale: {y_scale}")

  file_name = f"{FILE_NAME_MAP[x_axis]}_vs_{FILE_NAME_MAP[y_axis]}_{y_scale}.png"
  return os.path.join(OUTPUT_DIR, file_name)


def get_col_name(ax: AxType) -> str:
  """Returns the column name for the given AxType."""
  return COL_NAME_MAP[ax]


def get_ax_label(ax: AxType) -> str:
  """Returns the axis label for the given AxType."""
  return LABEL_MAP[ax]

def get_plot_id(x_axis: AxType) -> int:
  """Returns the plot_id for the given x_axis."""
  return PLOT_ID_MAP[x_axis]

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