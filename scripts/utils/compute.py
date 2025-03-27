import pandas as pd
import numpy as np
from utils.utils import AxType, get_fixed_param
import utils.config as cfg

def get_df(file: str, x_axis: AxType) -> pd.DataFrame:
  df = pd.read_csv(file)

  # Remove unnecessary text after the benchmark name
  df["name"] = df["name"].str.split("/", n=1).str[0]

  # Remove rows where corruption occured
  df = df[df["err_msg"].isna()]


  # Sort by benchmark name and FEC parameters
  df.sort_values(by=["name", "FEC"], ascending=[True, True], inplace=True)

  # Compute the message size & block size in KiB
  df["message_size_KiB"] = df["message_size_B"] // 1024
  df["block_size_KiB"] = df["block_size_B"] // 1024

  # Compute the encode times (and standard deviation) in ms
  df["time_ms"] = df["time_ns"] / 1e6
  df["time_ms_stddev"] = df["time_ns_stddev"] / 1e6

  # Compute the encode throughput (and standard deviation) in Gbps
  df["throughput_Gbps"] = (df["message_size_B"] * 8) / df["time_ns"] # equal to (#bits / 10^9) / (t_ns / 10^9) = #Gbits / s
  df["throughput_Gbps_stddev"] = df["throughput_Gbps"] * (df["time_ns_stddev"] / df["time_ns"]) # first order taylor expansion/error propagation

  for col in ["time_ms", "throughput_Gbps"]:
    df[f"{col}_err"] = cfg.Z_VALUE * (df[f"{col}_stddev"] / np.sqrt(df["num_iterations"]))

  # get only the rows with the fixed parameter:
  fixed_col, fixed_val = get_fixed_param(x_axis)
  df = df[df[fixed_col] == fixed_val]
  return df