import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from pyparsing import col
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
  "Xorec, Scalar",
  "Xorec, AVX",
  "Xorec, AVX2",
  "Xorec, AVX512",
]

XOREC_VARIANT_PLOT_ALGS = [
  "Xorec, AVX2",
  "Xorec (Unified Memory), AVX2",
  "Xorec (GPU Memory), AVX2"
]


def plot_EC(data: pd.DataFrame, y_type: PlotType, category: str, output_dir: str) -> None:
  """
  X-axis: EC parameters
  Y-Axis: throughput
  """
  assert(y_type in [PlotType.ENCODE, PlotType.DECODE]), f"Invalid y_type. Must be '{PlotType.ENCODE}' or '{PlotType.DECODE}'."
  assert(category in [Category.CPU, Category.GPU, Category.SIMD, Category.XOREC]), f"Invalid category. Must be '{Category.CPU}', '{Category.GPU}', '{Category.SIMD}', or '{Category.XOREC}'."

  # Filter the dataframe to only include the valid algorithms
  data = data[data[Column.NAME].isin(CATEGORY_INFO[category]["algorithms"])]

  assert(data[Column.IS_GPU_COMPUTE].nunique() == 1), "Error: The dataframe contains multiple GPU compute values. This is not expected."

  # Fix all parameters besides EC
  data = data[data[Column.DATA_SIZE] == FIXED_VALS[Column.DATA_SIZE]]
  data = data[data[Column.LOST_BLOCKS] == FIXED_VALS[Column.LOST_BLOCKS]]

  # Initialize labels, output file name, font-size, etc.
  output_file = os.path.join(output_dir, f"{CATEGORY_INFO[category]['file_prefix']}_ec_{y_type.value[0:3]}.pdf")
  x_label = r"Redundancy Ratio $(n/k)$"
  y_col = Column.ENC_THROUGHPUT if y_type == PlotType.ENCODE else Column.DEC_THROUGHPUT
  y_label = (
    "Encoding Throughput\n[Gbit/s]"
    if y_col == Column.ENC_THROUGHPUT
    else "Decoding Throughput\n[Gbit/s]"
  )

  plt.rcParams.update({"font.size": 17})
  fig, ax = plt.subplots(figsize=(14, 5))

  algorithms = data[Column.NAME].unique()
  categories = data[Column.EC].unique()
  x_label_loc = np.arange(len(categories))
  width = 0.15 #width of each bar
  tot_width = width * len(algorithms) #total width of each group of bars
  multiplier = 0

  for alg in algorithms:
    df_alg = data[data[Column.NAME] == alg]
    vals = df_alg[y_col].values
    ci_vals = df_alg[y_col + "_ci"].values
    bar_positions = x_label_loc + width * multiplier
    ax.bar(
      x=bar_positions,
      height=vals,
      width=width,
      label=alg,
      yerr=ci_vals,
      capsize=5,
    )

    for i, val in zip(bar_positions, vals):
      ax.text(
        x=i,
        y=val,
        s=f"{val:.0f}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        rotation=30,
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


def plot_datasize(data: pd.DataFrame, y_type: PlotType, category: str, output_dir: str) -> None:
  """
  X-axis: data size
  Y-Axis: throughput
  """
  assert(y_type in [PlotType.ENCODE, PlotType.DECODE]), f"Invalid y_type. Must be '{PlotType.ENCODE}' or '{PlotType.DECODE}'."
  assert(category in [Category.CPU, Category.GPU, Category.SIMD, Category.XOREC]), f"Invalid category. Must be '{Category.CPU}', '{Category.GPU}', '{Category.SIMD}', or '{Category.XOREC}'."

  # Filter the dataframe to only include the valid algorithms
  data = data[data[Column.NAME].isin(CATEGORY_INFO[category]["algorithms"])]

  assert(data[Column.IS_GPU_COMPUTE].nunique() == 1), "Error: The dataframe contains multiple GPU compute values. This is not expected."

  # Fix all parameters besides EC
  data = data[data[Column.EC] == FIXED_VALS[Column.EC]]
  data = data[data[Column.LOST_BLOCKS] == FIXED_VALS[Column.LOST_BLOCKS]]

  # Initialize labels, output file name, font-size, etc.
  output_file = os.path.join(output_dir, f"{CATEGORY_INFO[category]['file_prefix']}_datasize_{y_type.value[0:3]}.pdf")
  x_label = "Data Size [KiB]"
  y_col = Column.ENC_THROUGHPUT if y_type == PlotType.ENCODE else Column.DEC_THROUGHPUT
  y_label = (
    "Encoding Throughput\n[Gbit/s]"
    if y_col == Column.ENC_THROUGHPUT
    else "Decoding Throughput\n[Gbit/s]"
  )

  plt.rcParams.update({"font.size": 17})
  fig, ax = plt.subplots(figsize=(14, 5))

  algorithms = data[Column.NAME].unique()
  categories = data[Column.DATA_SIZE].unique()
  x_label_loc = np.arange(len(categories))
  width = 0.15 #width of each bar
  tot_width = width * len(algorithms) #total width of each group of bars
  multiplier = 0

  for alg in algorithms:
    df_alg = data[data[Column.NAME] == alg]
    vals = df_alg[y_col].values
    ci_vals = df_alg[y_col + "_ci"].values
    bar_positions = x_label_loc + width * multiplier


    ax.bar(
      x=bar_positions,
      height=vals,
      width=width,
      label=alg,
      yerr=ci_vals,
      capsize=5,
    )
    
    # Add the exact value on top of the bar
    for i, val in zip(bar_positions, vals):
      ax.text(
        x=i,
        y=val,
        s=f"{val:.0f}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        rotation=30,
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



def plot_ec_datasize_heatmap(data: pd.DataFrame, val_type: PlotType, category: str, output_dir: str) -> None:
  """
  X-axis: EC parameters
  Y-Axis: data size
  """
  assert(val_type in [PlotType.ENCODE, PlotType.DECODE]), f"Invalid y_type. Must be '{PlotType.ENCODE}' or '{PlotType.DECODE}'."
  assert(category in [Category.CPU]), f"Invalid category. Must be '{Category.CPU}'."

  # Filter the dataframe to only include the valid algorithms
  data = data[data[Column.NAME].isin(CATEGORY_INFO[category]["algorithms"])]

  assert(data[Column.IS_GPU_COMPUTE].nunique() == 1), "Error: The dataframe contains multiple GPU compute values. This is not expected."

  # Fix the number of lost blocks
  data = data[data[Column.LOST_BLOCKS] == FIXED_VALS[Column.LOST_BLOCKS]]

  # Initialize labels, output file name, font-size, etc.
  output_file = os.path.join(output_dir, f"{CATEGORY_INFO[category]['file_prefix']}_ec_datasize_heatmap_{val_type.value[0:3]}.pdf")
  x_label = r"Redundancy Ratio $(n/k)$"
  y_label = "Data Size [KiB]"
  val_col = Column.ENC_THROUGHPUT if val_type == PlotType.ENCODE else Column.DEC_THROUGHPUT

  algorithms = data[Column.NAME].unique()

  plt.rcParams.update({"font.size": 17})
  fig, axs = plt.subplots(1, len(algorithms), figsize=(len(algorithms)*7,7))
  axs = axs.flatten()
  
  # ensure gradients are on a global (not per-plot) scale
  global_vmin = data[val_col].min()
  global_vmax = data[val_col].max()

  for i, alg in enumerate(algorithms):
    df_alg = data[data[Column.NAME] == alg]


    pivot = df_alg.pivot(
        index=Column.DATA_SIZE,
        columns=Column.EC,
        values=val_col
      )
    
    ec_order = sorted(
      pivot.columns,
      key=lambda ec: (int(ec.strip("()").split("/")[1], 16), int(ec.strip("()").split("/")[0]))
    )
    pivot = pivot.reindex(columns=ec_order)
    
    im = axs[i].imshow(
      pivot.values,
      cmap="coolwarm",
      vmin=global_vmin,
      vmax=global_vmax,
    )

    for (y,x), val in np.ndenumerate(pivot.values):
      axs[i].text(
        x=x,
        y=y,
        s=f"{val:.0f}",
        ha="center",
        va="center",
        color="black",
        fontsize=12,
        fontweight="bold",
      )

    axs[i].set_xlabel(x_label)
    axs[i].set_ylabel(y_label)
    axs[i].set_xticks(np.arange(len(ec_order)))
    axs[i].set_xticklabels(ec_order, rotation=45)
    axs[i].set_yticks(np.arange(len(df_alg[Column.DATA_SIZE].unique())), labels=df_alg[Column.DATA_SIZE].unique())
    axs[i].set_title(f"{alg}")
    axs[i].invert_yaxis()


  plt.tight_layout()
  plt.savefig(output_file, format="pdf", dpi=300)
  plt.close(fig)

