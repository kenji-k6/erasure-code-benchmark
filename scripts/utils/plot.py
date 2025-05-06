import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from psutil.tests.test_posix import df
from utils.utils import Column, Category, CATEGORY_INFO, PlotType, ALGORITHMS, OUTPUT_DIR, FILE_PREFIX, FIXED_VALS
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
  assert(category in [Category.OPEN_SOURCE, Category.SIMD, Category.XOREC, Category.XOREC_GPU]), f"Invalid category. Must be '{Category.OPEN_SOURCE}', '{Category.SIMD}', or '{Category.XOREC}'."

  fixed_vals = CATEGORY_INFO[category][FIXED_VALS]

  # Filter the dataframe to only include the valid algorithms
  data = data[data[Column.NAME].isin(CATEGORY_INFO[category][ALGORITHMS])]

  if category == Category.XOREC or category == Category.XOREC_GPU:
    data = data[
      (data[Column.IS_GPU_COMPUTE] == False) |
      (
       (data[Column.IS_GPU_COMPUTE] == True) & 
       (data[Column.GPU_BLOCKS] == fixed_vals[Column.GPU_BLOCKS]) &
        (data[Column.THREADS_PER_BLOCK] == fixed_vals[Column.THREADS_PER_BLOCK] ) 
      ) 
    ]
    data = data[
      ((data[Column.CPU_THREADS] == fixed_vals[Column.CPU_THREADS]) & (data[Column.IS_GPU_COMPUTE] == False)) |
      ((data[Column.CPU_THREADS] ==  0) & (data[Column.IS_GPU_COMPUTE] == True))
    ]
  else:
    assert(data[Column.IS_GPU_COMPUTE].nunique() == 1), "Error: The dataframe contains multiple GPU compute values. This is not expected."
    data = data[data[Column.CPU_THREADS] == fixed_vals[Column.CPU_THREADS]]
  
  data = data[data[Column.MESSAGE_SIZE] == fixed_vals[Column.MESSAGE_SIZE]]
  data = data[data[Column.BLOCK_SIZE] == fixed_vals[Column.BLOCK_SIZE]]
  data = data[data[Column.LOST_BLOCKS] == fixed_vals[Column.LOST_BLOCKS]]
  

  # Initialize labels, output file name, font-size, etc.
  output_file = os.path.join(output_dir, f"{CATEGORY_INFO[category][FILE_PREFIX]}_ec_{y_type.value[0:3]}.pdf")
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
  if category == Category.XOREC:
    ax.set_yscale("log", base=10)
  ax.grid(axis="y", linestyle="--", which="both")

  # X-axis plotting
  ax.set_xlabel(x_label)
  ax.set_xticks(x_label_loc-(width/2)+(tot_width/2), labels=categories)

  ax.legend(loc=LEGEND_LOC, ncols=1, fontsize=LEGEND_FONTSIZE)

  plt.tight_layout()
  plt.savefig(output_file, format="pdf", dpi=300)
  plt.close(fig)


def plot_blocksize(data: pd.DataFrame, y_type: PlotType, category: Category, output_dir: str) -> None:
  """
  X-axis: block size
  Y-Axis: throughput
  """
  assert(y_type in [PlotType.ENCODE, PlotType.DECODE]), f"Invalid y_type. Must be '{PlotType.ENCODE}' or '{PlotType.DECODE}'."
  assert(category in [Category.OPEN_SOURCE, Category.SIMD, Category.XOREC]), f"Invalid category. Must be '{Category.OPEN_SOURCE}', '{Category.SIMD}', or '{Category.XOREC}'."

  fixed_vals = CATEGORY_INFO[category][FIXED_VALS]

  # Filter the dataframe to only include the valid algorithms
  data = data[data[Column.NAME].isin(CATEGORY_INFO[category][ALGORITHMS])]

  if category == Category.XOREC:
    data = data[
      (data[Column.IS_GPU_COMPUTE] == False) |
      (
       (data[Column.IS_GPU_COMPUTE] == True) & 
       (data[Column.GPU_BLOCKS] == fixed_vals[Column.GPU_BLOCKS]) &
        (data[Column.THREADS_PER_BLOCK] == fixed_vals[Column.THREADS_PER_BLOCK] ) 
      ) 
    ]
    data = data[
      ((data[Column.CPU_THREADS] == fixed_vals[Column.CPU_THREADS]) & (data[Column.IS_GPU_COMPUTE] == False)) |
      ((data[Column.CPU_THREADS] ==  0) & (data[Column.IS_GPU_COMPUTE] == True))
    ]
  else:
    assert(data[Column.IS_GPU_COMPUTE].nunique() == 1), "Error: The dataframe contains multiple GPU compute values. This is not expected."
    data = data[data[Column.CPU_THREADS] == fixed_vals[Column.CPU_THREADS]]

  data = data[data[Column.MESSAGE_SIZE] == fixed_vals[Column.MESSAGE_SIZE]]
  data = data[data[Column.EC] == fixed_vals[Column.EC]]
  data = data[data[Column.LOST_BLOCKS] == fixed_vals[Column.LOST_BLOCKS]]

  # Initialize labels, output file name, font-size, etc.
  output_file = os.path.join(output_dir, f"{CATEGORY_INFO[category][FILE_PREFIX]}_blocksize_{y_type.value[0:3]}.pdf")
  x_label = "Block Size [KiB]"
  y_col = Column.ENC_THROUGHPUT if y_type == PlotType.ENCODE else Column.DEC_THROUGHPUT
  y_label = (
    "Encoding Throughput\n[Gbit/s]"
    if y_col == Column.ENC_THROUGHPUT
    else "Decoding Throughput\n[Gbit/s]"
  )

  plt.rcParams.update({"font.size": MAIN_FONTSIZE})
  fig, ax = plt.subplots(figsize=(14, 5))

  algorithms = data[Column.NAME].unique()
  categories = data[Column.BLOCK_SIZE].unique()
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
  if category == Category.XOREC:
    ax.set_yscale("log", base=10)
  ax.grid(axis="y", linestyle="--", which="both")

  # X-axis plotting
  ax.set_xlabel(x_label)
  ax.set_xticks(x_label_loc-(width/2)+(tot_width/2), labels=categories)

  ax.legend(loc=LEGEND_LOC, ncols=1, fontsize=LEGEND_FONTSIZE)

  plt.tight_layout()
  plt.savefig(output_file, format="pdf", dpi=300)
  plt.close(fig)




def plot_cpu_threads(data: pd.DataFrame, y_type: PlotType, category: Category, output_dir: str, output_file_name = None) -> None:
  """
  X-axis: EC parameters
  Y-Axis: throughput
  """
  assert(y_type in [PlotType.ENCODE, PlotType.DECODE]), f"Invalid y_type. Must be '{PlotType.ENCODE}' or '{PlotType.DECODE}'."
  assert(category in [Category.PAPER_COMPARISON, Category.OPEN_SOURCE, Category.SIMD]), f"Invalid category. Must be '{Category.PAPER_COMPARISON}', '{Category.OPEN_SOURCE}' or '{Category.OPEN_SOURCE}'."

  fixed_vals = CATEGORY_INFO[category][FIXED_VALS]

  # Filter the dataframe to only include the valid algorithms
  data = data[data[Column.NAME].isin(CATEGORY_INFO[category][ALGORITHMS])]
  assert(data[Column.IS_GPU_COMPUTE].nunique() == 1), "Error: The dataframe contains multiple GPU compute values. This is not expected."

  # Fix all parameters besides EC
  data = data[data[Column.MESSAGE_SIZE] == fixed_vals[Column.MESSAGE_SIZE]]
  data = data[data[Column.BLOCK_SIZE] == fixed_vals[Column.BLOCK_SIZE]]
  data = data[data[Column.EC] == fixed_vals[Column.EC]]
  data = data[data[Column.LOST_BLOCKS] == fixed_vals[Column.LOST_BLOCKS]]

  # Initialize labels, output file name, font-size, etc.
  if output_file_name is None:
    output_file = os.path.join(output_dir, f"{CATEGORY_INFO[category][FILE_PREFIX]}_threads_{y_type.value[0:3]}.pdf")
  else:
    output_file = os.path.join(output_dir, output_file_name)

  x_label = r"Number of CPU Threads"
  y_col = Column.ENC_THROUGHPUT if y_type == PlotType.ENCODE else Column.DEC_THROUGHPUT
  y_label = (
    "Encoding Throughput\n[Gbit/s]"
    if y_col == Column.ENC_THROUGHPUT
    else "Decoding Throughput\n[Gbit/s]"
  )

  plt.rcParams.update({"font.size": MAIN_FONTSIZE})
  fig, ax = plt.subplots(figsize=(14, 5))

  algorithms = data[Column.NAME].unique()
  categories = data[Column.CPU_THREADS].unique()
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
  # ax.set_yscale("log", base=10)
  ax.grid(axis="y", linestyle="--", which="both")

  # X-axis plotting
  ax.set_xlabel(x_label)
  ax.set_xticks(x_label_loc-(width/2)+(tot_width/2), labels=categories)

  ax.legend(loc=LEGEND_LOC, ncols=1, fontsize=LEGEND_FONTSIZE)

  plt.tight_layout()
  plt.savefig(output_file, format="pdf", dpi=300)
  plt.close(fig)


def plot_lost_blocks(data: pd.DataFrame, y_type: PlotType, category: Category, output_dir: str) -> None:
  assert(y_type == PlotType.DECODE), f"Invalid y_type. Must be '{PlotType.DECODE}'."
  assert(category in [Category.OPEN_SOURCE, Category.XOREC]), f"Invalid category. Must be '{Category.OPEN_SOURCE}' or '{Category.XOREC}'."

  fixed_vals = CATEGORY_INFO[category][FIXED_VALS]
  # Filter the dataframe to only include the valid algorithms
  data = data[data[Column.NAME].isin(CATEGORY_INFO[category][ALGORITHMS])]
  # For XOR-EC, we only want the GPU compute, and CPU native version
  if category == Category.XOREC:
    data = data[
      data[Column.NAME].isin(["XOR-EC, AVX2","XOR-EC (GPU Computation)"])
    ]
  
  # Fix all parameters besides EC

  if category == Category.XOREC:
    data = data[
      (data[Column.IS_GPU_COMPUTE] == False) |
      (
       (data[Column.IS_GPU_COMPUTE] == True) & 
       (data[Column.GPU_BLOCKS] == fixed_vals[Column.GPU_BLOCKS]) &
        (data[Column.THREADS_PER_BLOCK] == fixed_vals[Column.THREADS_PER_BLOCK] ) 
      ) 
    ]
    data = data[
      ((data[Column.CPU_THREADS] == fixed_vals[Column.CPU_THREADS]) & (data[Column.IS_GPU_COMPUTE] == False)) |
      ((data[Column.CPU_THREADS] ==  0) & (data[Column.IS_GPU_COMPUTE] == True))
    ]
  else:
    assert(data[Column.IS_GPU_COMPUTE].nunique() == 1), "Error: The dataframe contains multiple GPU compute values. This is not expected."
    data = data[data[Column.CPU_THREADS] == fixed_vals[Column.CPU_THREADS]]
  
  data = data[data[Column.MESSAGE_SIZE] == fixed_vals[Column.MESSAGE_SIZE]]
  data = data[data[Column.BLOCK_SIZE] == fixed_vals[Column.BLOCK_SIZE]]
  data = data[data[Column.EC] == fixed_vals[Column.EC]]

  # Initialize labels, output file name, font-size, etc.
  output_file = os.path.join(output_dir, f"{CATEGORY_INFO[category][FILE_PREFIX]}_lost_{y_type.value[0:3]}.pdf")
  x_label = r"Lost blocks per chunk"
  y_col = Column.ENC_THROUGHPUT if y_type == PlotType.ENCODE else Column.DEC_THROUGHPUT
  y_label = (
    "Encoding Throughput\n[Gbit/s]"
    if y_col == Column.ENC_THROUGHPUT
    else "Decoding Throughput\n[Gbit/s]"
  )

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

  



def plot_paper_comparison(data: pd.DataFrame, data_paper: pd.DataFrame, y_type: PlotType, category: Category, output_dir: str) -> None:
  """
  X-axis: EC parameters
  Y-Axis: throughput
  """
  assert (y_type in [PlotType.ENCODE]), f"Invalid y_type. Must be '{PlotType.ENCODE}'."
  assert (category in [Category.PAPER_COMPARISON]), f"Invalid category. Must be '{Category.PAPER_COMPARISON}'."

  output_file_name_own = "cmp_threads_own.pdf"
  output_file_name_paper = "cmp_threads_paper.pdf"
  data = data.sort_values(by=[Column.NAME, Column.CPU_THREADS], ascending=[False, True])
  data_paper = data_paper.sort_values(by=[Column.NAME, Column.CPU_THREADS], ascending=[False, True])
  plot_cpu_threads(
    data=data,
    y_type=y_type,
    category=category,
    output_dir=output_dir,
    output_file_name=output_file_name_own
  )

  plot_cpu_threads(
    data=data_paper,
    y_type=y_type,
    category=category,
    output_dir=output_dir,
    output_file_name=output_file_name_paper
  )


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
    (4+8, 8), (4+16, 16), (8+16, 16), (4+32, 32), (8+32, 32)
  ]
  plot_P_recoverable_mds(output_dir, ec_params)
  plot_P_recoverable_xorec(output_dir, ec_params)

def plot_P_recoverable_mds(output_dir: str, ec_params) -> None:

  output_file = os.path.join(output_dir, "p_recoverable_mds.pdf")
  plt.rcParams.update({"font.size": 24})
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
      label=fr"MDS $({n}/{k})$",
      linestyle="-",
    )
  plt.xscale("log")
  plt.xlim((1e-3)*3, (1e-1)*1.1)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.yticks(np.arange(0, 1.01, 0.25))
  plt.legend(loc=LEGEND_LOC, ncols=1, fontsize=19)
  plt.grid(True, which="both", linestyle="--")
  plt.tight_layout()
  plt.savefig(output_file, format="pdf", dpi=300)
  plt.close()


def plot_P_recoverable_xorec(output_dir: str, ec_params) -> None:

  output_file = os.path.join(output_dir, "p_recoverable_xorec.pdf")
  plt.rcParams.update({"font.size": 24})
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
      label=fr"XOR-EC $({n}/{k})$",
      linestyle="-",
    )
  plt.xscale("log")
  plt.xlim((1e-3)*3, (1e-1)*1.1)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.yticks(np.arange(0, 1.01, 0.25))
  plt.legend(loc=LEGEND_LOC, ncols=1, fontsize=19)
  plt.grid(True, which="both", linestyle="--")
  plt.tight_layout()
  plt.savefig(output_file, format="pdf", dpi=300)
  plt.close()