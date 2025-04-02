import pandas as pd
import numpy as np
import utils.config as cfg

def get_df(file: str, plot_gpu: bool) -> pd.DataFrame:
  df = pd.read_csv(file)

  df  = df[df["is_gpu_bm"] == 1] if plot_gpu else df[df["is_gpu_bm"] == 0]

  # Remove unnecessary text after the benchmark name
  df["name"] = df["name"].str.split("/", n=1).str[0]

  # Remove rows where corruption occured
  df = df[df["err_msg"].isna()]

  df[["FEC_x", "FEC_y"]] = df["FEC"].str.extract(r'FEC\((\d+),(\d+)\)').astype(int)
  # Sort by benchmark name and FEC parameters
  df.sort_values(by=["name", "FEC_x", "FEC_y"], ascending=[True, True, True], inplace=True)

  # Compute the message size & block size in KiB
  df["message_size_KiB"] = df["message_size_B"] // 1024
  df["block_size_KiB"] = df["block_size_B"] // 1024

  # Compute the encode times (and standard deviation) in ms
  df["time_ms"] = df["time_ns"] / 1e6
  df["time_ms_stddev"] = df["time_ns_stddev"] / 1e6
  df["time_ms_err_min"] = df["time_ms"] - (df["time_ns_min"] / 1e6)
  df["time_ms_err_max"] = (df["time_ns_max"] / 1e6) - df["time_ms"]

  # Compute the encode throughput (and standard deviation) in Gbps
  df["throughput_Gbps"] = (df["message_size_B"] * 8) / df["time_ns"] # equal to (#bits / 10^9) / (t_ns / 10^9) = #Gbits / s
  df["throughput_Gbps_stddev"] = df["throughput_Gbps"] * (df["time_ns_stddev"] / df["time_ns"]) # first order taylor expansion/error propagation
  df["throughput_Gbps_err_min"] = df["throughput_Gbps"]-((df["message_size_B"] * 8) / df["time_ns_max"])
  df["throughput_Gbps_err_max"] = ((df["message_size_B"] * 8) / df["time_ns_min"])-df["throughput_Gbps"]

  for col in ["time_ms", "throughput_Gbps"]:
    df[f"{col}_ci"] = cfg.Z_VALUE * (df[f"{col}_stddev"] / np.sqrt(df["num_iterations"]))

  # If plotting for gpu result, make a nice string for the GPU parameters
  if plot_gpu:
    df["gpu_params"] = df.apply(lambda row: f"({row['num_gpu_blocks']}_{row['threads_per_gpu_block']})", axis=1)
  return df