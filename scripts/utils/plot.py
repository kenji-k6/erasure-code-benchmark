import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from psutil.tests.test_posix import df
from utils.utils import Column, Category, CATEGORY_INFO, PlotType
import math

MAIN_FONTSIZE = 15
LEGEND_FONTSIZE = 15
BAR_FONTSIZE = 12
LEGEND_LOC = "best"


def plot_EC(data: pd.DataFrame, y_type: PlotType, category: Category, output_dir: str) -> None:
  """
  X-axis: EC parameters
  Y-Axis: throughput
  """
  assert(y_type in [PlotType.ENCODE, PlotType.DECODE]), f"Invalid y_type. Must be '{PlotType.ENCODE}' or '{PlotType.DECODE}'."
  assert(category in [Category.CPU, Category.SIMD, Category.XOREC, Category.OPENSOURCE]), f"Invalid category. Must be '{Category.CPU}', '{Category.SIMD}', '{Category.OPENSOURCE}', or '{Category.XOREC}'."

  fixed_vals = CATEGORY_INFO[category]["fixed_vals"]

  # Filter the dataframe to only include the valid algorithms
  data = data[data[Column.NAME].isin(CATEGORY_INFO[category]["algorithms"])]

  if category == Category.XOREC:
    data = data[
      (data[Column.IS_GPU_COMPUTE] == False) |
      (
       (data[Column.IS_GPU_COMPUTE] == True) & 
       (data[Column.GPU_BLOCKS] == fixed_vals[Column.GPU_BLOCKS]) &
        (data[Column.THREADS_PER_BLOCK] == fixed_vals[Column.THREADS_PER_BLOCK] ) 
      ) 
    ]
  else:
    assert(data[Column.IS_GPU_COMPUTE].nunique() == 1), "Error: The dataframe contains multiple GPU compute values. This is not expected."

  # Fix all parameters besides EC
  data = data[data[Column.DATA_SIZE] == fixed_vals[Column.DATA_SIZE]]
  data = data[data[Column.LOST_BLOCKS] == fixed_vals[Column.LOST_BLOCKS]]

  # Initialize labels, output file name, font-size, etc.
  output_file = os.path.join(output_dir, f"{CATEGORY_INFO[category]['file_prefix']}_ec_{y_type.value[0:3]}.pdf")
  x_label = r"Redundancy $(n/k)$"
  y_col = Column.ENC_THROUGHPUT if y_type == PlotType.ENCODE else Column.DEC_THROUGHPUT
  y_label = (
    "Encoding Throughput\n[Gbit/s]"
    if y_col == Column.ENC_THROUGHPUT
    else "Decoding Throughput\n[Gbit/s]"
  )

  plt.rcParams.update({"font.size": MAIN_FONTSIZE})
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
        fontsize=BAR_FONTSIZE,
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

  ax.legend(loc=LEGEND_LOC, ncols=1, fontsize=LEGEND_FONTSIZE)

  plt.tight_layout()
  plt.savefig(output_file, format="pdf", dpi=300)
  plt.close(fig)


def plot_datasize(data: pd.DataFrame, y_type: PlotType, category: Category, output_dir: str) -> None:
  """
  X-axis: data size
  Y-Axis: throughput
  """
  assert(y_type in [PlotType.ENCODE, PlotType.DECODE]), f"Invalid y_type. Must be '{PlotType.ENCODE}' or '{PlotType.DECODE}'."
  assert(category in [Category.CPU, Category.SIMD, Category.XOREC, Category.OPENSOURCE]), f"Invalid category. Must be '{Category.CPU}', '{Category.SIMD}', '{Category.OPENSOURCE}', or '{Category.XOREC}'."

  fixed_vals = CATEGORY_INFO[category]["fixed_vals"]

  # Filter the dataframe to only include the valid algorithms
  data = data[data[Column.NAME].isin(CATEGORY_INFO[category]["algorithms"])]

  if category == Category.XOREC:
    data = data[
      (data[Column.IS_GPU_COMPUTE] == False) |
      (
       (data[Column.IS_GPU_COMPUTE] == True) & 
       (data[Column.GPU_BLOCKS] == fixed_vals[Column.GPU_BLOCKS]) &
        (data[Column.THREADS_PER_BLOCK] == fixed_vals[Column.THREADS_PER_BLOCK] ) 
      ) 
    ]
  else:
    assert(data[Column.IS_GPU_COMPUTE].nunique() == 1), "Error: The dataframe contains multiple GPU compute values. This is not expected."

  # Fix all parameters besides EC
  data = data[data[Column.EC] == fixed_vals[Column.EC]]
  data = data[data[Column.LOST_BLOCKS] == fixed_vals[Column.LOST_BLOCKS]]

  # Initialize labels, output file name, font-size, etc.
  output_file = os.path.join(output_dir, f"{CATEGORY_INFO[category]['file_prefix']}_datasize_{y_type.value[0:3]}.pdf")
  x_label = "Data Size [KiB]"
  y_col = Column.ENC_THROUGHPUT if y_type == PlotType.ENCODE else Column.DEC_THROUGHPUT
  y_label = (
    "Encoding Throughput\n[Gbit/s]"
    if y_col == Column.ENC_THROUGHPUT
    else "Decoding Throughput\n[Gbit/s]"
  )

  plt.rcParams.update({"font.size": MAIN_FONTSIZE})
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
        fontsize=BAR_FONTSIZE,
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

  ax.legend(loc=LEGEND_LOC, ncols=1, fontsize=LEGEND_FONTSIZE)

  plt.tight_layout()
  plt.savefig(output_file, format="pdf", dpi=300)
  plt.close(fig)

def plot_lost_blocks(data: pd.DataFrame, y_type: PlotType, category: Category, output_dir: str) -> None:
  """
  X-axis: lost blocks
  Y-Axis: throughput
  """
  assert(y_type in [PlotType.DECODE]), f"Invalid y_type. Must be '{PlotType.DECODE}'."
  assert(category in [Category.CPU, Category.SIMD, Category.XOREC, Category.OPENSOURCE]), f"Invalid category. Must be '{Category.CPU}', '{Category.SIMD}', '{Category.OPENSOURCE}', or '{Category.XOREC}'."

  fixed_vals = CATEGORY_INFO[category]["fixed_vals"]
  
  # Filter the dataframe to only include the valid algorithms
  data = data[data[Column.NAME].isin(CATEGORY_INFO[category]["algorithms"])]

  if category == Category.XOREC:
    data = data[
      (data[Column.IS_GPU_COMPUTE] == False) |
      (
       (data[Column.IS_GPU_COMPUTE] == True) & 
       (data[Column.GPU_BLOCKS] == fixed_vals[Column.GPU_BLOCKS]) &
        (data[Column.THREADS_PER_BLOCK] == fixed_vals[Column.THREADS_PER_BLOCK] ) 
      ) 
    ]
  else:
    assert(data[Column.IS_GPU_COMPUTE].nunique() == 1), "Error: The dataframe contains multiple GPU compute values. This is not expected."
  
  # Fix all parameters besides lost blocks
  data = data[data[Column.DATA_SIZE] == fixed_vals[Column.DATA_SIZE]]
  data = data[data[Column.EC] == fixed_vals[Column.EC]]

  output_file = os.path.join(output_dir, f"{CATEGORY_INFO[category]['file_prefix']}_lostblocks_{y_type.value[0:3]}.pdf")
  x_label = "Lost Blocks"
  y_col = Column.DEC_THROUGHPUT
  y_label = "Decoding Throughput\n[Gbit/s]"

  plt.rcParams.update({"font.size": MAIN_FONTSIZE})
  fig, ax = plt.subplots(figsize=(14, 5))

  algorithms = data[Column.NAME].unique()
  categories = data[Column.LOST_BLOCKS].unique()
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
        fontsize=BAR_FONTSIZE,
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
  ax.legend(loc=LEGEND_LOC, ncols=1, fontsize=LEGEND_FONTSIZE)
  plt.tight_layout()
  plt.savefig(output_file, format="pdf", dpi=300)
  plt.close(fig)


  


def plot_ec_datasize_heatmap(data: pd.DataFrame, val_type: PlotType, category: Category, output_dir: str) -> None:
  """
  X-axis: EC parameters
  Y-Axis: data size
  """
  assert(val_type in [PlotType.ENCODE, PlotType.DECODE]), f"Invalid y_type. Must be '{PlotType.ENCODE}' or '{PlotType.DECODE}'."
  assert(category in [Category.CPU]), f"Invalid category. Must be '{Category.CPU}'."

  fixed_vals = CATEGORY_INFO[category]["fixed_vals"]

  # Filter the dataframe to only include the valid algorithms
  data = data[data[Column.NAME].isin(CATEGORY_INFO[category]["algorithms"])]

  assert(data[Column.IS_GPU_COMPUTE].nunique() == 1), "Error: The dataframe contains multiple GPU compute values. This is not expected."

  # Fix the number of lost blocks
  data = data[data[Column.LOST_BLOCKS] == fixed_vals[Column.LOST_BLOCKS]]

  # Initialize labels, output file name, font-size, etc.
  output_file = os.path.join(output_dir, f"{CATEGORY_INFO[category]['file_prefix']}_ec_datasize_heatmap_{val_type.value[0:3]}.pdf")
  x_label = r"Redundancy $(n/k)$"
  y_label = "Data Size [KiB]"
  val_col = Column.ENC_THROUGHPUT if val_type == PlotType.ENCODE else Column.DEC_THROUGHPUT

  algorithms = data[Column.NAME].unique()

  plt.rcParams.update({"font.size": MAIN_FONTSIZE})
  fig, axs = plt.subplots(2, (len(algorithms)+2-1)//2, figsize=(((len(algorithms)+2-1)//2)*7,2*5))
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
        fontsize=MAIN_FONTSIZE,
        fontweight="bold",
      )

    axs[i].set_xlabel(x_label, fontsize=MAIN_FONTSIZE)
    axs[i].set_ylabel(y_label, fontsize=MAIN_FONTSIZE)
    axs[i].set_xticks(np.arange(len(ec_order)))
    axs[i].set_xticklabels(ec_order, rotation=45)
    axs[i].set_yticks(np.arange(len(df_alg[Column.DATA_SIZE].unique())), labels=df_alg[Column.DATA_SIZE].unique())
    axs[i].set_title(f"{alg}")
    axs[i].invert_yaxis()

  plt.tight_layout()
  plt.savefig(output_file, format="pdf", dpi=300)
  plt.close(fig)



# Below are functions to plot probabilities of recoverability

#auxiliary functions
def P_recoverable_MDS(k: int, m: int, P_drop: float) -> float:
  """
  Calculate the probability of failure for MDS codes.
  """
  P_failure = 0.0
  for i in range(0,m+1):
    P_failure += math.comb(k+m, i) * (P_drop**i) * ((1-P_drop)**(k+m-i))
  return P_failure

def P_recoverable_Xorec(k: int, m: int, P_drop: float) -> float:
  """
  Calculate the probability of failure for XOREC codes.
  """
  assert(k % m == 0), "k must be divisible by m for XOREC codes."
  return ((1+((P_drop * k)/m))**m) * ((1 - P_drop)**k)

def plot_P_recoverable(output_dir: str) -> None:
  ec_params = [
    (4+32, 32), (8+64, 64), (16+128, 128)
  ]
  plot_P_recoverable_mds(output_dir, ec_params)
  plot_P_recoverable_xorec(output_dir, ec_params)

def plot_P_recoverable_mds(output_dir: str, ec_params) -> None:

  output_file = os.path.join(output_dir, "p_recoverable_mds.pdf")
  plt.rcParams.update({"font.size": 21})
  P_drop = np.logspace(-3, 0, 100)
  plt.figure(figsize=(7, 5))

  x_label = r"$P_{\mathrm{drop}}$"
  y_label = r"$P^{\mathrm{MDS}}_{\mathrm{rec}}$"

  for (n, k) in ec_params:
    m = n - k
    P_MDS = [P_recoverable_MDS(k, m, p) for p in P_drop]
    plt.plot(
      P_drop,
      P_MDS,
      label=fr"MDS ${n}/{k}$",
      linestyle="-",
    )
  plt.xscale("log")
  plt.xlim((1e-3)*3, (1e-1)*1.1)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.yticks(np.arange(0, 1.01, 0.25))
  plt.legend(loc=LEGEND_LOC, ncols=1, fontsize=LEGEND_FONTSIZE)
  plt.grid(True, which="both", linestyle="--")
  plt.tight_layout()
  plt.savefig(output_file, format="pdf", dpi=300)
  plt.close()


def plot_P_recoverable_xorec(output_dir: str, ec_params) -> None:

  output_file = os.path.join(output_dir, "p_recoverable_xorec.pdf")
  plt.rcParams.update({"font.size": 21})
  P_drop = np.logspace(-3, 0, 100)
  plt.figure(figsize=(7, 5))

  x_label = r"$P_{\mathrm{drop}}$"
  y_label = r"$P^{\mathrm{XOR\!-\!EC}}_{\mathrm{rec}}$"

  for (n, k) in ec_params:
    m = n - k
    P_Xorec = [P_recoverable_Xorec(k, m, p) for p in P_drop]
    plt.plot(
      P_drop,
      P_Xorec,
      label=fr"Xorec ${n}/{k}$",
      linestyle="-",
    )
  plt.xscale("log")
  plt.xlim((1e-3)*3, (1e-1)*1.1)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.yticks(np.arange(0, 1.01, 0.25))
  plt.legend(loc=LEGEND_LOC, ncols=1, fontsize=LEGEND_FONTSIZE)
  plt.grid(True, which="both", linestyle="--")
  plt.tight_layout()
  plt.savefig(output_file, format="pdf", dpi=300)
  plt.close()