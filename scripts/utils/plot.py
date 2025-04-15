import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from utils.utils import Column, FIXED_VALS



def plot_EC(df: pd.DataFrame, tp_type: str, output_dir: str) -> None:
  """
  X-axis: EC parameters
  Y-Axes: throughput
  """
  assert(tp_type in ["encode", "decode"]), "Invalid tp_type. Must be 'encode' or 'decode'."
  y_col = Column.ENC_THROUGHPUT if tp_type == "encode" else Column.DEC_THROUGHPUT

  # Fix all parameters besides EC
  df = df[df[Column.BLOCK_SIZE] == FIXED_VALS[Column.BLOCK_SIZE]]
  df = df[df[Column.LOST_BLOCKS] == FIXED_VALS[Column.LOST_BLOCKS]]

  # Initialize labels, output file name, font-size, etc.
  output_file = os.path.join(output_dir, f"ec_{tp_type[0:3]}.pdf")
  x_label = r"Redundancy Ratio $(n/k)$"
  y_label = (
    "Encoding Throughput\n[Gbit/s]"
    if y_col == Column.ENC_THROUGHPUT
    else "Decoding Throughput\n[Gbit/s]"
  )

  plt.rcParams.update({"font.size": 17})
  fig, ax = plt.subplots(figsize=(14, 5))

  algorithms = df[Column.NAME].unique()
  categories = df[Column.EC].unique()
  x_label_loc = np.arange(len(categories))
  width = 0.15 #width of each bar
  tot_width = width * len(algorithms) #total width of each group of bars
  multiplier = 0

  for alg in algorithms:
    df_alg = df[df[Column.NAME] == alg]
    vals = df_alg[y_col].values
    ci_vals = df_alg[y_col + "_ci"].values

    offset = width * multiplier
    ax.bar(
      x=x_label_loc + offset,
      height=vals,
      width=width,
      label=alg,
      yerr=ci_vals,
      capsize=5,
    )
    multiplier += 1

  # Y-axis plotting
  ax.set_ylabel(y_label)
  ax.set_yscale("log", base=10)
  ax.grid(axis="y", linestyle="--", which="both")

  # X-axis plotting
  ax.set_xlabel(x_label)
  ax.set_xticks(x_label_loc-(width/2)+(tot_width/2), labels=categories)

  ax.legend(loc="upper right", ncols=1)

  plt.tight_layout()
  plt.savefig(output_file, format="pdf", dpi=300)
  plt.close(fig)
  pass