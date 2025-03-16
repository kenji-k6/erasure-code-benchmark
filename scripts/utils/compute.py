import pandas as pd
import numpy as np
from typing import Dict, List
from utils.config import Z_VALUE

def get_grouped_dataframes(inp_file: str, algs: List[str]) -> Dict[int, pd.DataFrame]:
  df = pd.read_csv(inp_file)


  df["name"] = df["name"].str.split("/", n=1).str[0]
  df = df[df["err_msg"].isna()]

  if algs:
    pattern = '|'.join(algs)
    df = df[df["name"].str.contains(pattern, na=False)]

  df.sort_values(by="name", ascending=True, inplace=True)

  df["encode_time_ms"] = df["encode_time_ns"] / 1e6
  df["decode_time_ms"] = df["decode_time_ns"] / 1e6
  df["tot_data_size_KiB"] = df["tot_data_size_B"] // 1024


  for col in ["encode_time_ns", "decode_time_ns", "encode_throughput_Gbps", "decode_throughput_Gbps"]:
    df[f"{col}_lower"] = Z_VALUE * (df[f"{col}_stddev"] / np.sqrt(df["num_iterations"]))
    df[f"{col}_upper"] = Z_VALUE * (df[f"{col}_stddev"] / np.sqrt(df["num_iterations"]))

  for col in ["encode_time", "decode_time"]:
    df.rename(columns={f"{col}_ns_lower": f"{col}_ms_lower", f"{col}_ns_upper": f"{col}_ms_upper"}, inplace=True)
    df[f"{col}_ms_lower"] = df[f"{col}_ms_lower"] / 1e6
    df[f"{col}_ms_upper"] = df[f"{col}_ms_upper"] / 1e6
  
  dfs = {plot_id: group for plot_id, group in df.groupby("plot_id")}
  return dfs