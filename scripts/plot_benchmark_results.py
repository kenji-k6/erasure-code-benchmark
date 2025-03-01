import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np

# File / directory paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "../results/raw/benchmark_results.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../results/processed/")

def ensure_output_directory():
  """Ensure the output directory exists."""
  os.makedirs(OUTPUT_DIR, exist_ok=True)



def plot_scatter(dfs: dict[int, pd.DataFrame], x_col: str, y_col: str, x_label: str, y_label: str, file_name: str, plot_id: int):
  """Helper function to generate and save scatter plots."""
  df = dfs[plot_id]

  sns.set_theme(style="whitegrid")
  plt.figure(figsize=(10, 6))

  sns.scatterplot(data=df, x=x_col, y=y_col, hue="name", palette="tab10")
  plt.xlabel(x_label, fontsize=12)
  plt.ylabel(y_label, fontsize=12)
  plt.xscale("log")
  plt.yscale("linear")
  plt.legend(title="Libraries", bbox_to_anchor=(1.05, 1), loc="upper left")
  plt.title(get_plot_title(dfs, plot_id), fontsize=12)

  ax = plt.gca()
  x_ticks = sorted(df[x_col].unique())
  ax.set_xticks(x_ticks)
  ax.set_xticklabels([f"{x}" for x in x_ticks])


  # Plot the confidence intervals
  for name, group in df.groupby("name"):
    plt.errorbar(group[x_col], group[y_col],
                 yerr=[group[f"{y_col}_lower"], group[f"{y_col}_upper"]],
                 fmt='o', color=ax.get_legend().legend_handles[df["name"].unique().tolist().index(name)].get_color())

  plt.tight_layout()
  plt.savefig(os.path.join(OUTPUT_DIR, file_name))
  plt.close()

  

def get_plot_title(dfs: dict[int, pd.DataFrame], plot_id:int):
  """Generate a plot title containing all the constant parameters."""
  df = dfs[plot_id]
  num_data_blocks = df.iloc[0]["num_data_blocks"]
  redundancy_ratio = df.iloc[0]["redundancy_ratio"]
  num_lost_blocks = df.iloc[0]["num_lost_blocks"]
  num_iterations = df.iloc[0]["num_iterations"]
  buffer_size_mib = df.iloc[0]["tot_data_size_MiB"]

  if plot_id == 0:
    return f"#Data Blocks: {num_data_blocks}, #Redundancy Ratio: {redundancy_ratio}, #Lost Blocks: {num_lost_blocks}, #Iterations: {num_iterations}"
  elif plot_id == 1:
    return f"Buffer Size: {buffer_size_mib} MiB, #Data Blocks: {num_data_blocks}, #Lost Blocks: {num_lost_blocks}, #Iterations: {num_iterations}" 
  elif plot_id == 2:
    return f"Buffer Size: {buffer_size_mib} MiB, #Data Blocks: {num_data_blocks}, #Parity Blocks = {int(num_data_blocks*redundancy_ratio)}, #Iterations: {num_iterations}"
  
  return "ERROR: Invalid plot_id"



if __name__ == "__main__":
  ensure_output_directory()
  df = pd.read_csv(INPUT_FILE)

  # Clean up 'name' column by removing the '/iterations:...' part
  df["name"] = df["name"].str.split("/", n=1).str[0]

  # Remove rows where corruption occured
  df = df[df["err_msg"].isna()]

  # Get additional columns 
  df["time_ms"] = df["time_ns"] / 1e6
  df["encode_time_ms"] = df["encode_time_ns"] / 1e6
  df["decode_time_ms"] = df["decode_time_ns"] / 1e6
  df["tot_data_size_MiB"] = df["tot_data_size_B"] // (1024 * 1024)
  df["throughput_Gbps"] = (df["tot_data_size_B"] * 8 / 1e9) / (df["time_ns"] / 1e9)


  # Compute the confidence  intervals per row for the throughput and time columns
  # The formula used is CI = mean +/- z * (stddev / sqrt(n))
  z = 1.96 # 95% confidence interval

  df["encode_time_ms_lower"] = (df["encode_time_ns"] - z * (df["encode_time_stddev_ns"] / np.sqrt(df["num_iterations"]))) / 1e6
  df["encode_time_ms_upper"] = (df["encode_time_ns"] + z * (df["encode_time_stddev_ns"] / np.sqrt(df["num_iterations"]))) / 1e6

  df["decode_time_ms_lower"] = (df["decode_time_ns"] - z * (df["decode_time_stddev_ns"] / np.sqrt(df["num_iterations"]))) / 1e6
  df["decode_time_ms_upper"] = (df["decode_time_ns"] + z * (df["decode_time_stddev_ns"] / np.sqrt(df["num_iterations"]))) / 1e6

  df["encode_throughput_Gbps_lower"] = (df["encode_throughput_Gbps"] - z * (df["encode_throughput_stddev_Gbps"] / np.sqrt(df["num_iterations"])))
  df["encode_throughput_Gbps_upper"] = (df["encode_throughput_Gbps"] + z * (df["encode_throughput_stddev_Gbps"] / np.sqrt(df["num_iterations"])))

  df["decode_throughput_Gbps_lower"] = (df["decode_throughput_Gbps"] - z * (df["decode_throughput_stddev_Gbps"] / np.sqrt(df["num_iterations"])))
  df["decode_throughput_Gbps_upper"] = (df["decode_throughput_Gbps"] + z * (df["decode_throughput_stddev_Gbps"] / np.sqrt(df["num_iterations"])))

  
  
  dfs = {plot_id: group for plot_id, group in df.groupby("plot_id")}

  # X: Buffer Size, Y: Encode/Decode Time
  plot_scatter(dfs, "tot_data_size_MiB", "encode_time_ms", "Buffer Size (MiB)", "Encode Time (ms)", "buffersize_vs_encodetime.png", 0)
  plot_scatter(dfs, "tot_data_size_MiB", "decode_time_ms", "Buffer Size (MiB)", "Decode Time (ms)", "buffersize_vs_decodetime.png", 0)

  # X: Buffer Size, Y: Throughput
  # plot_scatter(dfs, "tot_data_size_MiB", "throughput_Gbps", "Buffer Size (MiB)", "Throughput (Gbps)", "buffer_size_vs_throughput.png", 0)


  # # X: Num Parity Blocks, Y: Time
  # plot_scatter(dfs, "num_parity_blocks", "time_ms", "#Parity Blocks", "Time (ms)", "parity_blocks_vs_time.png", 1)

  # # X: Num Parity Blocks, Y: Throughput
  # plot_scatter(dfs, "num_parity_blocks", "throughput_Gbps", "#Parity Blocks", "Throughput (Gbps)", "parity_blocks_vs_throughput.png", 1)


  # # X: Num Lost Blocks, Y: Time
  # plot_scatter(dfs, "num_lost_blocks", "time_ms", "#Lost Blocks", "Time (ms)", "lost_blocks_vs_time.png", 2)

  # # X: Num Lost Blocks, Y: Throughput
  # plot_scatter(dfs, "num_lost_blocks", "throughput_Gbps", "#Lost Blocks", "Throughput (Gbps)", "lost_blocks_vs_throughput.png", 2)