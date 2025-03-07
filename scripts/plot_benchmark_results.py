import os
import argparse
from webbrowser import get
import cpuinfo
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any
from collections import namedtuple
from enum import Enum

# File / directory paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = "undefined"
LIN_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../results/processed/linear/")
LOG_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../results/processed/log/")

# AVX XOR / AVX2 XOR max throughput
AVX_XOR_CPI_XEON = 0.33
AVX2_XOR_CPI_XEON = 0.33
AVX_BITS = 128
AVX2_BITS = 256
AVX_COLOR = "cornflowerblue"
AVX2_COLOR = "indigo"

CPUInfo = namedtuple("CPUInfo", ["model_name", "clock_speed_GHz", "l1_cache_size_KiB", "l2_cache_size_KiB", "l3_cache_size_KiB"])

class AxType(Enum):
  ENCODE_T = 0
  DECODE_T = 1
  ENCODE_TP = 2
  DECODE_TP = 3
  BUF_SIZE = 4
  PARITY_BLKS = 5
  LOST_BLKS = 6



def ensure_output_directories() -> None:
  """Ensure the output directory exists."""
  os.makedirs(LIN_OUTPUT_DIR, exist_ok=True)
  os.makedirs(LOG_OUTPUT_DIR, exist_ok=True)

def get_cpu_info() -> CPUInfo:
  """Get CPU information."""
  info = cpuinfo.get_cpu_info()
  model_name = info["brand_raw"]

  #clock speed in GHz 
  clock_speed_GHz = float(psutil.cpu_freq().max if psutil.cpu_freq().max else "0") / 1e3

  #L1, l2, l3 Cache sizes
  l1_cache_size = int(info.get("l1_data_cache_size", "0")) // 1024
  l2_cache_size = int(info.get("l2_cache_size", "0")) // 1024
  l3_cache_size = int(info.get("l3_cache_size", "0")) // 1024

  return CPUInfo(model_name=model_name,
                 clock_speed_GHz=clock_speed_GHz,
                 l1_cache_size_KiB=l1_cache_size,
                 l2_cache_size_KiB=l2_cache_size,
                 l3_cache_size_KiB=l3_cache_size)

def get_plot_title(df: pd.DataFrame, plot_id: int, cpu_info: CPUInfo) -> str:
  """Generate a  plot title containing all the constant parameters."""
  first_row = df.iloc[0]
  cpu_title = f"CPU: {cpu_info.model_name}, Max clock frequency: {cpu_info.clock_speed_GHz} GHz"

  titles = {
    0: f"#Data Blocks: {first_row['num_data_blocks']}, #Redundancy Ratio: {first_row['redundancy_ratio']}, #Lost Blocks: {first_row['num_lost_blocks']}, #Iterations: {first_row['num_iterations']}\n{cpu_title}",
    1: f"Buffer Size: {first_row['tot_data_size_KiB']} KiB, #Data Blocks: {first_row['num_data_blocks']}, #Lost Blocks: {first_row['num_lost_blocks']}, #Iterations: {first_row['num_iterations']}\n{cpu_title}",
    2: f"Buffer Size: {first_row['tot_data_size_KiB']} KiB, #Data Blocks: {first_row['num_data_blocks']}, #Parity Blocks: {int(first_row['num_parity_blocks'])}, #Iterations: {first_row['num_iterations']}\n{cpu_title}"
  }

  return titles.get(plot_id, "ERROR: Invalid plot_id")

def get_output_path(x_ax: AxType, y_ax: AxType, y_scale: str) -> str:
  """Generate the file name for the plot."""
  ax_map = {
    AxType.ENCODE_T: "encodetime",
    AxType.DECODE_T: "decodetime",
    AxType.ENCODE_TP: "encodethroughput",
    AxType.DECODE_TP: "decodethroughput",
    AxType.BUF_SIZE: "buffersize",
    AxType.PARITY_BLKS: "parityblocks",
    AxType.LOST_BLKS: "lostblocks"
  }

  x_part = ax_map[x_ax]
  y_part = ax_map[y_ax]
  file_name = f"{x_part}_vs_{y_part}_{y_scale}.png"
  output_dir = LOG_OUTPUT_DIR if (y_scale == "log") else LIN_OUTPUT_DIR
  return os.path.join(output_dir, file_name)

def get_col_name(ax: AxType) -> str:
  """Return the column name in the dataframe for the given AxType."""
  ax_map = {
    AxType.ENCODE_T: "encode_time_ms",
    AxType.DECODE_T: "decode_time_ms",
    AxType.ENCODE_TP: "encode_throughput_Gbps",
    AxType.DECODE_TP: "decode_throughput_Gbps",
    AxType.BUF_SIZE: "tot_data_size_KiB",
    AxType.PARITY_BLKS: "num_parity_blocks",
    AxType.LOST_BLKS: "num_lost_blocks"
  }
  return ax_map[ax]

def get_ax_label(ax: AxType) -> str:
  """Return the axis label for the given AxType."""
  ax_map = {
    AxType.ENCODE_T: "Encode Time (ms)",
    AxType.DECODE_T: "Decode Time (ms)",
    AxType.ENCODE_TP: "Encode Throughput (Gbps)",
    AxType.DECODE_TP: "Decode Throughput (Gbps)",
    AxType.BUF_SIZE: "Buffer Size",
    AxType.PARITY_BLKS: "#Parity Blocks",
    AxType.LOST_BLKS: "#Lost Blocks"
  }
  return ax_map[ax]

def get_min_encode_time(xor_cpi: float, avx_bits: int, cpu_frequency_GHz: float, data_size_B: int, num_data_blocks: int, num_parity_blocks: int, num_lost_data_blocks: int) -> float:
  """Compute the theoretical minimum encoding time (single threaded)."""
  block_size_bits = (data_size_B // num_data_blocks) * 8

  num_xor = (num_data_blocks * block_size_bits) / avx_bits
  num_cycles = num_xor * xor_cpi
  encode_time_ms = (num_cycles / cpu_frequency_GHz) / 1e6 # equal to (#cycles / (#GHz * 1e9)) * 1e3  = (#cycles / #Hz) * 1e3
  return encode_time_ms


def get_max_encode_throughput(xor_cpi: float, avx_bits: int, cpu_frequency_GHz: float, data_size_B: int, num_data_blocks: int, num_parity_blocks: int, num_lost_data_blocks: int) -> float:
  """Compute the theoretical maximum encoding throughput (single threaded)."""
  encode_time_ms = get_min_encode_time(xor_cpi, avx_bits, cpu_frequency_GHz, data_size_B, num_data_blocks, num_parity_blocks, num_lost_data_blocks)
  data_size_bits = data_size_B * 8
  encode_throughput_Gbps = (data_size_bits / 1e6) / encode_time_ms # equal to (#bits/1e9)/(#ms/1e3) = #Gbps
  return encode_throughput_Gbps


def get_min_decode_time(xor_cpi: float, avx_bits: int, cpu_frequency_GHz: float, data_size_B: int, num_data_blocks: int, num_parity_blocks: int, num_lost_data_blocks: int) -> float:
  """Compute the theoretical minimum decoding time (single threaded)."""
  block_size_bits = (data_size_B // num_data_blocks) * 8
  blocks_per_parity_block = num_data_blocks // num_parity_blocks
  num_xors_per_lost_data_block = ((blocks_per_parity_block) * block_size_bits) / avx_bits
  tot_num_xors = num_xors_per_lost_data_block * num_lost_data_blocks
  num_cycles = tot_num_xors * xor_cpi
  decode_time_ms = (num_cycles / cpu_frequency_GHz) / 1e6 # equal to (#cycles / (#GHz * 1e9)) * 1e3  = (#cycles / #Hz) * 1e3
  return decode_time_ms


def get_max_decode_throughput(xor_cpi: float, avx_bits: int, cpu_frequency_GHz: float, data_size_B: int, num_data_blocks: int, num_parity_blocks: int, num_lost_data_blocks: int) -> float:
  """Compute the theoretical maximum decoding throughput (single threaded)."""
  decode_time_ms = get_min_decode_time(xor_cpi, avx_bits, cpu_frequency_GHz, data_size_B, num_data_blocks, num_parity_blocks, num_lost_data_blocks)
  data_size_bits = data_size_B * 8
  decode_throughput_Gbps = (data_size_bits / 1e6) / decode_time_ms # equal to (#bits/1e9)/(#ms/1e3) = #Gbps
  return decode_throughput_Gbps

avx_avx2_func = {
  AxType.ENCODE_T: get_min_encode_time,
  AxType.DECODE_T: get_min_decode_time,
  AxType.ENCODE_TP: get_max_encode_throughput,
  AxType.DECODE_TP: get_max_decode_throughput
}

def plot_cache_sizes(cpu_info: CPUInfo) -> None:
  l2_cache_size = cpu_info.l2_cache_size_KiB
  l3_cache_size = cpu_info.l3_cache_size_KiB
  plt.axvline(x=l2_cache_size, color="red", linestyle="--", label="L2 Cache Size\n" + f"({l2_cache_size} KiB)")
  plt.axvline(x=l3_cache_size, color="green", linestyle="--", label="L3 Cache Size\n" + f"({l3_cache_size} KiB)")

def plot_xticks(df: pd.DataFrame, x_ax: AxType) -> None:
  ax = plt.gca()
  x_col = get_col_name(x_ax)
  x_ticks = sorted(df[x_col].unique())

  if x_ax == AxType.BUF_SIZE:
    x_ticklabels = [(lambda x: f"{x//1024} MiB" if x >= 1024 else f"{x} KiB")(x) for x in x_ticks]
    rot = 45
  else:
    x_ticklabels = [f"{x}" for x in x_ticks]
    rot = 0
  ax.set_xticks(ticks=x_ticks, labels=x_ticklabels, rotation=rot)
  ax.set_xlim(left=x_ticks[0]/2, right=x_ticks[-1]*2)

def plot_confidence_intervals(df: pd.DataFrame, x_ax: AxType, y_ax: AxType) -> None:
  x_col = get_col_name(x_ax)
  y_col = get_col_name(y_ax)
  low_err_col = f"{y_col}_lower"
  up_err_col = f"{y_col}_upper"
  ax = plt.gca()
  for name, group in df.groupby("name"):
    plt.errorbar(group[x_col], group[y_col], yerr=[group[low_err_col].to_numpy(), group[up_err_col].to_numpy()],
                 fmt='o', color=ax.get_legend().legend_handles[df["name"].unique().tolist().index(name)].get_color(), capsize=5 
    )

def plot_avx_avx2_xor(df: pd.DataFrame, x_ax: AxType, y_ax: AxType, cpu_frequency_GHz: float) -> None:
  x_col = get_col_name(x_ax)
  y_col = get_col_name(y_ax)
  f = avx_avx2_func[y_ax]

  params_df = df[[
    "tot_data_size_KiB",
    "num_data_blocks",
    "num_parity_blocks",
    "num_lost_blocks"
  ]].drop_duplicates().sort_values(by=x_col, ascending=True)

  x_values = params_df[x_col].unique()

  y_values_AVX = params_df.apply(
    lambda row: f(AVX_XOR_CPI_XEON, AVX_BITS, cpu_frequency_GHz, row["tot_data_size_KiB"]*1024,
                  row["num_data_blocks"], row["num_parity_blocks"], row["num_lost_blocks"]
                  ),
    axis=1
  ).tolist()

  y_values_AVX2 = params_df.apply(
    lambda row: f(AVX2_XOR_CPI_XEON, AVX2_BITS, cpu_frequency_GHz, row["tot_data_size_KiB"]*1024,
                  row["num_data_blocks"], row["num_parity_blocks"], row["num_lost_blocks"]
                  ),
    axis=1
  ).tolist()

  assert(len(x_values) == len(y_values_AVX) == len(y_values_AVX2))
  plt.plot(x_values, y_values_AVX, label="AVX XOR", color=AVX_COLOR, linestyle="--")
  plt.plot(x_values, y_values_AVX2, label="AVX2 XOR", color=AVX2_COLOR, linestyle="--")


def write_scatter_plot(dfs: Dict[int, pd.DataFrame], x_ax: AxType, y_ax: AxType, y_scale: str, plot_id: int) -> None:
  """Generate and save scatter plot."""
  output_file = get_output_path(x_ax, y_ax, y_scale)
  cpu_info = get_cpu_info()
  df = dfs[plot_id]

  x_label= get_ax_label(x_ax)
  y_label = get_ax_label(y_ax)
  x_col = get_col_name(x_ax)
  y_col = get_col_name(y_ax)

  sns.set_theme(style="whitegrid")
  plt.figure(figsize=(10, 6))
  sns.scatterplot(data=df, x=x_col, y=y_col, hue="name", palette="tab10")

  plt.xlabel(x_label, fontsize=12)
  plt.ylabel(y_label, fontsize=12)

  plt.xscale("log", base=2)
  plt.yscale(y_scale)

  if x_ax == AxType.BUF_SIZE: plot_cache_sizes(cpu_info)
  plot_avx_avx2_xor(df, x_ax, y_ax, cpu_info.clock_speed_GHz)

  plt.legend(title="Libraries", bbox_to_anchor=(1.05, 1), loc="upper left")
  plt.title(get_plot_title(df, plot_id, cpu_info), fontsize=12)

  plot_xticks(df, x_ax)
  plot_confidence_intervals(df, x_ax, y_ax)

  plt.tight_layout()
  plt.savefig(output_file, dpi=300)
  plt.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Plot benchmark results")
  parser.add_argument("--input-file", type=str, help="Input file name (in ../results/raw/ directory)", required=True)
  args = parser.parse_args()
  INPUT_FILE = os.path.join(SCRIPT_DIR, f"../results/raw/{args.input_file}")


  ensure_output_directories()
  df = pd.read_csv(INPUT_FILE)

  # Process and clean data
  df["name"] = df["name"].str.split("/", n=1).str[0]
  df = df[df["err_msg"].isna()]

  # Sort by name
  df.sort_values(by="name", ascending=True, inplace=True)

  # Compute additional columns 
  df["encode_time_ms"] = df["encode_time_ns"] / 1e6
  df["decode_time_ms"] = df["decode_time_ns"] / 1e6
  df["tot_data_size_KiB"] = df["tot_data_size_B"] // 1024


  # Compute confidence intervals
  # CI = mean +/- z * (stddev / sqrt(n))
  z = 3.291 # 99.9% confidence interval

  for col in ["encode_time_ns", "decode_time_ns", "encode_throughput_Gbps", "decode_throughput_Gbps"]:
    df[f"{col}_lower"] = z * (df[f"{col}_stddev"] / np.sqrt(df["num_iterations"]))
    df[f"{col}_upper"] = z * (df[f"{col}_stddev"] / np.sqrt(df["num_iterations"]))

  # Convert the time Confidence interval columns to milliseconds
  for col in ["encode_time", "decode_time"]:
    df.rename(columns={f"{col}_ns_lower": f"{col}_ms_lower", f"{col}_ns_upper": f"{col}_ms_upper"}, inplace=True)
    df[f"{col}_ms_lower"] = df[f"{col}_ms_lower"] / 1e6
    df[f"{col}_ms_upper"] = df[f"{col}_ms_upper"] / 1e6


  dfs = {plot_id: group for plot_id, group in df.groupby("plot_id")}

  plot_params = [
    # X: Buffer Size, Y: Encode/Decode Time
    (AxType.BUF_SIZE, AxType.ENCODE_T, "linear", 0),
    (AxType.BUF_SIZE, AxType.DECODE_T, "linear", 0),
    (AxType.BUF_SIZE, AxType.ENCODE_T, "log", 0),
    (AxType.BUF_SIZE, AxType.DECODE_T, "log", 0),

    # X: Buffer Size, Y: Encode/Decode Throughput
    (AxType.BUF_SIZE, AxType.ENCODE_TP, "linear", 0),
    (AxType.BUF_SIZE, AxType.DECODE_TP, "linear", 0),
    (AxType.BUF_SIZE, AxType.ENCODE_TP, "log", 0),
    (AxType.BUF_SIZE, AxType.DECODE_TP, "log", 0),


    # X: Num Parity Blocks, Y: Encode/Decode Time
    (AxType.PARITY_BLKS, AxType.ENCODE_T, "linear", 1),
    (AxType.PARITY_BLKS, AxType.DECODE_T, "linear", 1),
    (AxType.PARITY_BLKS, AxType.ENCODE_T, "log", 1),
    (AxType.PARITY_BLKS, AxType.DECODE_T, "log", 1),

    # X: Num Parity Blocks, Y: Encode/Decode Throughput
    (AxType.PARITY_BLKS, AxType.ENCODE_TP, "linear", 1),
    (AxType.PARITY_BLKS, AxType.DECODE_TP, "linear", 1),
    (AxType.PARITY_BLKS, AxType.ENCODE_TP, "log", 1),
    (AxType.PARITY_BLKS, AxType.DECODE_TP, "log", 1),


    # X: Num Lost Blocks, Y: Encode/Decode Time
    (AxType.LOST_BLKS, AxType.ENCODE_T, "linear", 2),
    (AxType.LOST_BLKS, AxType.DECODE_T, "linear", 2),
    (AxType.LOST_BLKS, AxType.ENCODE_T, "log", 2),
    (AxType.LOST_BLKS, AxType.DECODE_T, "log", 2),

    # X: Num Lost Blocks, Y: Encode/Decode Throughput
    (AxType.LOST_BLKS, AxType.ENCODE_TP, "linear", 2),
    (AxType.LOST_BLKS, AxType.DECODE_TP, "linear", 2),
    (AxType.LOST_BLKS, AxType.ENCODE_TP, "log", 2),
    (AxType.LOST_BLKS, AxType.DECODE_TP, "log", 2)
  ]

  for params in plot_params:
    write_scatter_plot(dfs, *params)
  