import pandas as pd
import numpy as np
from typing import Tuple
from utils.utils import Z_VALUE

def parse_ec_str(ec: str) -> Tuple[int, int]:
  ec = ec.strip("()") # remove brackets
  x_str, y_str = ec.split("/")
  return int(x_str), int(y_str)

def get_df(path: str) -> pd.DataFrame:
  """
  Reads a CSV file, parses data, computes additional columns, and returns a DataFrame
  """
  df = pd.read_csv(path)
      
  # Ensure that no measurements have corruption, delete the error column afterwards
  assert(df["err_msg"].isna().all()), "Error: Some measurements have corruption"

  # Change gpu col-type to bool
  df["gpu_computation"] = df["gpu_computation"].astype(bool)

  # Sort the Dataframe by:
  # name -> gpu_blocks -> threads_per_block -> block_size -> lost_blocks -> EC
  # introduce auxiliary columns for EC sorting
  df[["EC_x", "EC_y"]] = df["EC"].apply(lambda x: pd.Series(parse_ec_str(x)))
  df = df.sort_values(
    by=[
      "name",
      "gpu_blocks",
      "threads_per_block",
      "message_size_B",
      "lost_blocks",
      "EC_y",
      "EC_x",
    ],
    axis=0,
    ascending=[True, True, True, True, True, True, True],
    ignore_index=True,
  )

  # Drop the auxiliary columns
  df = df.drop(columns=["EC_x", "EC_y"])


  # Get KiB for the block size
  df["block_size_KiB"] = (df["block_size_B"] // 1024).astype(int)
  print(df.columns)
  # Get KiB for the data size
  df["message_size_KiB"] = (df["message_size_B"] // 1024).astype(int)

  # Compute the 99.9% confidence interval(s) for later plotting
  for col in ["encode_throughput_Gbps", "decode_throughput_Gbps"]:
    df[f"{col}_ci"] = Z_VALUE * (df[f"{col}_stddev"] / np.sqrt(df["iterations"]))



  # Drop the unneeded columns
  df = df.drop(
    columns=[
      "err_msg",
      "iterations",
      "warmup_iterations", 
      "message_size_B",
      "block_size_B",
      "encode_time_ns",
      "encode_time_ns_stddev",
      "decode_time_ns",
      "decode_time_ns_stddev",
      "encode_throughput_Gbps_stddev",
      "decode_throughput_Gbps_stddev",
    ],
    axis=1
  )

  return df

