import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from typing import Dict

# File / directory paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "../results/raw/benchmark_results.csv")
LIN_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../results/processed/linear/")
LOG_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../results/processed/log/")


def ensure_output_directories() -> None:
  """Ensure the output directory exists."""
  os.makedirs(LIN_OUTPUT_DIR, exist_ok=True)
  os.makedirs(LOG_OUTPUT_DIR, exist_ok=True)

def get_plot_title(df: pd.DataFrame, plot_id: int) -> str:
  """Generate a  plot title containing all the constant parameters."""
  first_row = df.iloc[0]
  titles = {
    0: f"#Data Blocks: {first_row['num_data_blocks']}, #Redundancy Ratio: {first_row['redundancy_ratio']}, #Lost Blocks: {first_row['num_lost_blocks']}, #Iterations: {first_row['num_iterations']}",
    1: f"Buffer Size: {first_row['tot_data_size_KiB']} KiB, #Data Blocks: {first_row['num_data_blocks']}, #Lost Blocks: {first_row['num_lost_blocks']}, #Iterations: {first_row['num_iterations']}",
    2: f"Buffer Size: {first_row['tot_data_size_KiB']} KiB, #Data Blocks: {first_row['num_data_blocks']}, #Parity Blocks: {int(first_row['num_parity_blocks'])}, #Iterations: {first_row['num_iterations']}"
  }
  return titles.get(plot_id, "ERROR: Invalid plot_id")


def make_scatter_plot(dfs: Dict[int, pd.DataFrame], x_col: str, y_col: str, x_label: str, y_label: str, y_scale: str, file_name: str, plot_id: int) -> None:
  """Generate and save scatter plots."""
  df = dfs[plot_id]
  sns.set_theme(style="whitegrid")
  plt.figure(figsize=(10, 6))
  output_directory = LIN_OUTPUT_DIR if y_scale == "linear" else LOG_OUTPUT_DIR


  sns.scatterplot(data=df, x=x_col, y=y_col, hue="name", palette="tab10")
  plt.xlabel(x_label, fontsize=12)
  plt.ylabel(y_label, fontsize=12)
  plt.xscale("log", base=2)
  plt.yscale(y_scale)
  plt.legend(title="Libraries", bbox_to_anchor=(1.05, 1), loc="upper left")
  plt.title(get_plot_title(df, plot_id), fontsize=12)  

  # Set proper x-ticks
  ax = plt.gca()
  x_ticks = sorted(df[x_col].unique())
  if (x_col == "tot_data_size_KiB"):
    x_ticklabels = [(lambda x: f"{x//1024} MiB" if x >= 1024 else f"{x} KiB")(x) for x in x_ticks]
    rot = 45
  else:
    x_ticklabels = [f"{x}" for x in x_ticks]
    rot = 0
  ax.set_xticks(ticks=x_ticks, labels=x_ticklabels, rotation=rot)

  ax.set_xlim(left=x_ticks[0]/2, right=x_ticks[-1]*2)
  
  # Plot confidence intervals (has to be done with a loop, to ensure the correct color)
  for name, group in df.groupby("name"):
    plt.errorbar(group[x_col], group[y_col],
                 yerr=[group[f"{y_col}_lower"].to_numpy(), group[f"{y_col}_upper"].to_numpy()],
                 fmt='o', color=ax.get_legend().legend_handles[df["name"].unique().tolist().index(name)].get_color(), capsize=5
                 )
  plt.tight_layout()
  plt.savefig(os.path.join(output_directory, file_name))
  plt.close()



if __name__ == "__main__":
  ensure_output_directories()
  df = pd.read_csv(INPUT_FILE)

  # Process and clean data
  df["name"] = df["name"].str.split("/", n=1).str[0]
  df = df[df["err_msg"].isna()]


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

  # Define plot parameters
  plot_params = [
    # Linear Y-scale
    # X: Buffer Size, Y: Encode/Decode Time
    ("tot_data_size_KiB", "encode_time_ms", "Buffer Size", "Encode Time (ms)", "linear", "buffersize_vs_encodetime_linear.png", 0),
    ("tot_data_size_KiB", "decode_time_ms", "Buffer Size", "Decode Time (ms)", "linear", "buffersize_vs_decodetime_linear.png", 0),
    # X: Buffer Size, Y: Encode/Decode Throughput
    ("tot_data_size_KiB", "encode_throughput_Gbps", "Buffer Size", "Encode Throughput (Gbps)", "linear", "buffersize_vs_encodethroughput_linear.png", 0),
    ("tot_data_size_KiB", "decode_throughput_Gbps", "Buffer Size", "Decode Throughput (Gbps)", "linear", "buffersize_vs_decodethroughput_linear.png", 0),

    # X: Num Parity Blocks, Y: Encode/Decode Time
    ("num_parity_blocks", "encode_time_ms", "#Parity Blocks", "Encode Time (ms)", "linear", "parityblocks_vs_encodetime_linear.png", 1),
    ("num_parity_blocks", "decode_time_ms", "#Parity Blocks", "Decode Time (ms)", "linear", "parityblocks_vs_decodetime_linear.png", 1),
    # X: Num Parity Blocks, Y: Encode/Decode Throughput
    ("num_parity_blocks", "encode_throughput_Gbps", "#Parity Blocks", "Encode Throughput (Gbps)", "linear", "parityblocks_vs_encodethroughput_linear.png", 1),
    ("num_parity_blocks", "decode_throughput_Gbps", "#Parity Blocks", "Decode Throughput (Gbps)", "linear", "parityblocks_vs_decodethroughput_linear.png", 1),

    # X: Num Lost Blocks, Y: Encode/Decode Time
    ("num_lost_blocks", "encode_time_ms", "#Lost Blocks", "Encode Time (ms)", "linear", "lostblocks_vs_encodetime_linear.png", 2),
    ("num_lost_blocks", "decode_time_ms", "#Lost Blocks", "Decode Time (ms)", "linear", "lostblocks_vs_decodetime_linear.png", 2),
    # X: Num Lost Blocks, Y: Encode/Decode Throughput
    ("num_lost_blocks", "encode_throughput_Gbps", "#Lost Blocks", "Encode Throughput (Gbps)", "linear", "lostblocks_vs_encodethroughput_linear.png", 2),
    ("num_lost_blocks", "decode_throughput_Gbps", "#Lost Blocks", "Decode Throughput (Gbps)", "linear", "lostblocks_vs_decodethroughput_linear.png", 2),


    # Log Y-scale
    # X: Buffer Size, Y: Encode/Decode Time
    ("tot_data_size_KiB", "encode_time_ms", "Buffer Size", "Encode Time (ms)", "log", "buffersize_vs_encodetime_log.png", 0),
    ("tot_data_size_KiB", "decode_time_ms", "Buffer Size", "Decode Time (ms)", "log", "buffersize_vs_decodetime_log.png", 0),
    # X: Buffer Size, Y: Encode/Decode Throughput
    ("tot_data_size_KiB", "encode_throughput_Gbps", "Buffer Size", "Encode Throughput (Gbps)", "log", "buffersize_vs_encodethroughput_log.png", 0),
    ("tot_data_size_KiB", "decode_throughput_Gbps", "Buffer Size", "Decode Throughput (Gbps)", "log", "buffersize_vs_decodethroughput_log.png", 0),

    # X: Num Parity Blocks, Y: Encode/Decode Time
    ("num_parity_blocks", "encode_time_ms", "#Parity Blocks", "Encode Time (ms)", "log", "parityblocks_vs_encodetime_log.png", 1),
    ("num_parity_blocks", "decode_time_ms", "#Parity Blocks", "Decode Time (ms)", "log", "parityblocks_vs_decodetime_log.png", 1),
    # X: Num Parity Blocks, Y: Encode/Decode Throughput
    ("num_parity_blocks", "encode_throughput_Gbps", "#Parity Blocks", "Encode Throughput (Gbps)", "log", "parityblocks_vs_encodethroughput_log.png", 1),
    ("num_parity_blocks", "decode_throughput_Gbps", "#Parity Blocks", "Decode Throughput (Gbps)", "log", "parityblocks_vs_decodethroughput_log.png", 1),

    # X: Num Lost Blocks, Y: Encode/Decode Time
    ("num_lost_blocks", "encode_time_ms", "#Lost Blocks", "Encode Time (ms)", "log", "lostblocks_vs_encodetime_log.png", 2),
    ("num_lost_blocks", "decode_time_ms", "#Lost Blocks", "Decode Time (ms)", "log", "lostblocks_vs_decodetime_log.png", 2),
    # X: Num Lost Blocks, Y: Encode/Decode Throughput
    ("num_lost_blocks", "encode_throughput_Gbps", "#Lost Blocks", "Encode Throughput (Gbps)", "log", "lostblocks_vs_encodethroughput_log.png", 2),
    ("num_lost_blocks", "decode_throughput_Gbps", "#Lost Blocks", "Decode Throughput (Gbps)", "log", "lostblocks_vs_decodethroughput_log.png", 2)
  ]

  # Generate plots
  for params in plot_params:
    make_scatter_plot(dfs, *params)