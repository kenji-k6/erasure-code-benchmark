import os
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
from typing import Tuple
from collections import namedtuple
BASE_DIR = "logs-long-run"
PLOT_DIR = "plots"

DirnameConfig = namedtuple("DirnameConfig", ["proc", "packet_size", "fq", "fq_num", "trial"])
ClientLogData = namedtuple("ClientLogData", ["dropped", "total", "drop_rate"])


def parse_server_snmp_log(dirname: str) -> Tuple[int, int]:
  """
  Parses the server_snmp.log file, to check
  if there are discrepancies to the client logs
  """
  filepath = os.path.join(BASE_DIR, dirname, "server_snmp.log")
  with open(filepath, "r") as f:
    lines = f.readlines()
  
  if len(lines) != 10:
    raise ValueError(f"Invalid number of lines in server_snmp.log: {len(lines)}")
  
  initial_RcvBufErrors = int(lines[2].split(" ")[5])
  final_RcvBufErrors = int(lines[7].split(" ")[5])

  return initial_RcvBufErrors, final_RcvBufErrors

def is_valid_trial(dirname: str) -> bool:
  """
  Checks if the trial is valid by comparing the initial and final RcvBufErrors on the server.
  """
  initial_RcvBufErrors, final_RcvBufErrors = parse_server_snmp_log(dirname)
  diff = final_RcvBufErrors - initial_RcvBufErrors
  return diff == 0




def parse_dirname_config(dirname: str) -> DirnameConfig:
  """
  Parses the directory name to extract configuration parameters.
  """
  pattern = r"proc(\d+)-buf(\d+).*?(nofq|fq)(?:-(\d+)G)?-trial(\d+)"
  match = re.search(pattern, dirname)

  if match:
    return DirnameConfig(
      proc=int(match.group(1)),
      packet_size=int(match.group(2)),
      fq=match.group(3) == "fq",
      fq_num=int(match.group(4)) if match.group(4) else -1,
      trial=int(match.group(5))
    )
  else:
    raise ValueError(f"Failed to extract configuration from directory name ({dirname})")


def parse_client_iperf_log(dirname: str) -> ClientLogData:
  """
  Parses the client_iperf.log file to extract dropped and total packets from the receiver line.
  """
  pattern = r"(\d+)/(\d+)\s+\(\d+(\.\d+)?%\)"
  filepath = os.path.join(BASE_DIR, dirname, "client_iperf.log")

  with open(filepath, "r") as f:
    lines = f.readlines()
  
  for line in reversed(lines):
    if "receiver" in line:
      match = re.search(pattern, line)

      if match:
        dropped = int(match.group(1))
        total = int(match.group(2))
        drop_rate = (dropped / total)
        return ClientLogData(
          dropped=dropped,
          total=total,
          drop_rate=drop_rate
        )
  raise ValueError(f"Failed to extract client_iperf.log data ({dirname})")

def get_dataframe() -> pd.DataFrame:
  """
  Returns a DataFrame with the parsed data from the logs.
  """
  rows = []
  for dirname in os.listdir(BASE_DIR):
    dirpath = os.path.join(BASE_DIR, dirname)
    if os.path.isdir(dirpath):
      dirname_config = parse_dirname_config(dirname)
      client_log_data = parse_client_iperf_log(dirname)
      valid = is_valid_trial(dirname)
      rows.append(
        {
          "proc": dirname_config.proc,
          "packet_size_KiB": dirname_config.packet_size//1024,
          "fq": dirname_config.fq,
          "fq_num": dirname_config.fq_num,
          "trial": dirname_config.trial,
          "dropped": client_log_data.dropped,
          "total": client_log_data.total,
          "drop_rate": client_log_data.drop_rate,
          "valid": valid
        }
      )
  return pd.DataFrame(rows)

def write_plot(df: pd.DataFrame) -> None:
  """
  Creates the actual plots
  """
  # Sanity checks
  assert len(df["proc"].unique()) == 1, "More than one proc type in the dataframe"
  assert len(df["fq"].unique()) == 1, "More than one fq type in the dataframe"
  assert len(df["fq_num"].unique()) == 1, "More than one fq_num in the dataframe"


  # Get x and y axis labels
  x_label = r"$P_{drop}$"
  y_label = "UDP payload\nsize [KiB]"

  # Get x and y axis columns
  x_col = "drop_rate"
  y_col = "packet_size_KiB"

  # Sort the dataframe by trial and packet size
  df = df.sort_values(
    by=["trial", y_col],
    ascending=[True, True],
    inplace=False)
  
  # Set the font size to 15
  plt.rcParams.update({"font.size": 17})

  # Create the plot
  plt.figure(figsize=(10, 2.5))

  # Colors for the measurements
  x_log = np.log10(df[x_col])
  norm = mcolors.Normalize(vmin=x_log.min(), vmax=x_log.max())
  cmap = mcolors.LinearSegmentedColormap.from_list(
    "green_red", ["lawngreen","limegreen", "darkorange", "red", "darkred"]
  )
  # cmap = cm.get_cmap("RdYlGn_r")
  colors = cmap(norm(x_log))

  # Plot the data points
  plt.scatter(
    x=df[x_col],
    y=df[y_col],
    c=colors,
    s=50,
    alpha=1.0,
    edgecolors="none"
  )
    
  ax = plt.gca()
  plt.xscale("log", base=10)
  plt.xlabel(x_label)

  plt.yscale("log", base=2)
  plt.yticks(
    ticks = df[y_col].unique()
  )
  ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
  plt.ylabel(y_label)

   # Add minor ticks to the x-axis
  ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))  # Adjust the number of minor ticks as needed
  plt.minorticks_on()  # Enable minor ticks


  plt.grid(axis="x", linestyle="--", alpha=1.0, which="both")
  plt.grid(axis="y", linestyle="--", alpha=1.0, which="both")
  ax.set_axisbelow(True)
  plt.tight_layout()
  plt.savefig(
    os.path.join(PLOT_DIR, f"drop_rates_cscs.pdf"),
    format="pdf",
    dpi=300
  )
  plt.close()


def main() -> None:
  df = get_dataframe()
  # Remove invalid trials
  df = df[df["valid"]]
  unique_combs = df[["proc", "fq", "fq_num"]].drop_duplicates()

  final_params = {
    "proc": 16,
    "fq": True,
    "fq_num": 10
  }

  filtered_df = df[
    (df["proc"] == final_params["proc"]) &
    (df["fq"] == final_params["fq"]) &
    (df["fq_num"] == final_params["fq_num"])
  ]



  df1 = filtered_df[filtered_df["packet_size_KiB"] == 1]
  df2 = filtered_df[filtered_df["packet_size_KiB"] == 2]
  df4 = filtered_df[filtered_df["packet_size_KiB"] == 4]
  df8 = filtered_df[filtered_df["packet_size_KiB"] == 8]

  min1 = df1["drop_rate"].min()
  min2 = df2["drop_rate"].min()
  min4 = df4["drop_rate"].min()
  min8 = df8["drop_rate"].min()

  max1 = df1["drop_rate"].max()
  max2 = df2["drop_rate"].max()
  max4 = df4["drop_rate"].max()
  max8 = df8["drop_rate"].max()

  mean1 = df1["drop_rate"].mean()
  mean2 = df2["drop_rate"].mean()
  mean4 = df4["drop_rate"].mean()
  mean8 = df8["drop_rate"].mean()

  print(f"DF1, Min: {min1}", f"Max: {max1}", f"Mean: {mean1}")
  print(f"DF2, Min: {min2}", f"Max: {max2}", f"Mean: {mean2}")
  print(f"DF4, Min: {min4}", f"Max: {max4}", f"Mean: {mean4}")
  print(f"DF8, Min: {min8}", f"Max: {max8}", f"Mean: {mean8}")

  # Create the plot
  write_plot(filtered_df)
  

  
if __name__ == "__main__":
  main()
