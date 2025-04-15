import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from utils.utils import Column, FIXED_VALS, Category, CATEGORY_INFO, PlotType



CPU_PLOT_ALGS = [
  "Xorec, AVX2",
  "ISA-L",
  "CM256",
  "LEOPARD"
]

GPU_PLOT_ALGS = [
  "Xorec (GPU Computation)"
]

XOREC_SIMD_PLOT_ALGS = [
  "Xorec, AVX",
  "Xorec, AVX2",
  "Xorec, AVX512",
]

XOREC_VARIANT_PLOT_ALGS = [

]


def plot_EC(df: pd.DataFrame, y_type: str, category: str, output_dir: str) -> None:
  """
  X-axis: EC parameters
  Y-Axes: throughput
  """
  assert(y_type in [PlotType.ENCODE, PlotType.DECODE]), f"Invalid y_type. Must be '{PlotType.ENCODE}' or '{PlotType.DECODE}'."
  assert(category in [Category.CPU, Category.GPU, Category.SIMD, Category.XOREC]), f"Invalid category. Must be '{Category.CPU}', '{Category.GPU}', '{Category.SIMD}', or '{Category.XOREC}'."
  
  # Filter the dataframe to only include the valid algorithms
  df = df[df[Column.NAME].isin(CATEGORY_INFO[category]["algorithms"])]

  assert(df[Column.IS_GPU_COMPUTE].nunique() == 1), "Error: The dataframe contains multiple GPU compute values. This is not expected."

  # Fix all parameters besides EC
  df = df[df[Column.BLOCK_SIZE] == FIXED_VALS[Column.BLOCK_SIZE]]
  df = df[df[Column.LOST_BLOCKS] == FIXED_VALS[Column.LOST_BLOCKS]]

  # Initialize labels, output file name, font-size, etc.
  output_file = os.path.join(output_dir, f"{CATEGORY_INFO[category]['file_prefix']}_ec_{y_type[0:3]}.pdf")
  x_label = r"Redundancy Ratio $(n/k)$"
  y_col = Column.ENC_THROUGHPUT if y_type == "encode" else Column.DEC_THROUGHPUT
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


def plot_ec_blocksize_heatmap(df: pd.DataFrame, y_type: str, category: str, output_dir: str) -> None:
  """
  X-axis: EC parameters
  Y-Axis: block size
  """

  assert(y_type in [PlotType.ENCODE, PlotType.DECODE]), f"Invalid y_type. Must be '{PlotType.ENCODE}' or '{PlotType.DECODE}'."
  assert(category in [Category.CPU]), f"Invalid category. Must be '{Category.CPU}'."

  # Filter the dataframe to only include the valid algorithms
  df = df[df[Column.NAME].isin(CATEGORY_INFO[category]["algorithms"])]

  assert(df[Column.IS_GPU_COMPUTE].nunique() == 1), "Error: The dataframe contains multiple GPU compute values. This is not expected."

  # Fix the number of lost blocks
  df = df[df[Column.LOST_BLOCKS] == FIXED_VALS[Column.LOST_BLOCKS]]

  # Initialize labels, output file name, font-size, etc.
  output_file = os.path.join(output_dir, f"{CATEGORY_INFO[category]['file_prefix']}_ec_blocksize_heatmap_{y_type[0:3]}.pdf")

#TODO: Continue Heatmap from here!!!
