import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.hardware_info import CPUInfo, get_cpu_info
from utils.utils import AxType, get_output_path, get_col_name, get_ax_label, get_plot_title
import utils.config as cfg


def plot_cache_sizes(cpu_info: CPUInfo) -> None:
  pass
  # plt.axvline(x=cpu_info.l1_cache_size_KiB, color="blue", linestyle="--", label="L1 Cache Size")
  # plt.axvline(x=cpu_info.l2_cache_size_KiB, color="red", linestyle="--", label="L2 Cache Size")
  # plt.axvline(x=cpu_info.l3_cache_size_KiB, color="green", linestyle="--", label="L3 Cache Size")

def get_plot_df(df: pd.DataFrame, x_axis: AxType) -> pd.DataFrame:
  if x_axis == AxType.BLK_SIZE: 
    plot_df = df[df[get_col_name(AxType.CPU_THREADS)] == cfg.FIXED_CPU_THREADS]
    plot_df = plot_df[plot_df[get_col_name(AxType.FEC_RATIO)] == cfg.FIXED_FEC_RATIO]
  elif x_axis == AxType.FEC_RATIO:
    plot_df = df[df[get_col_name(AxType.CPU_THREADS)] == cfg.FIXED_CPU_THREADS]
    plot_df = plot_df[plot_df[get_col_name(AxType.BLK_SIZE)] == cfg.FIXED_BLOCK_SIZE]
  elif x_axis == AxType.CPU_THREADS:
    plot_df = df[df[get_col_name(AxType.FEC_RATIO)] == cfg.FIXED_FEC_RATIO]
    plot_df = plot_df[plot_df[get_col_name(AxType.BLK_SIZE)] == cfg.FIXED_BLOCK_SIZE]
  else:
    raise ValueError(f"Invalid x_axis value: {x_axis}. Must be BLK_SIZE, FEC_RATIO, or CPU_THREADS.")
  
  return plot_df
  



def write_cpu_plot(
  df: pd.DataFrame,
  x_axis: AxType,
  y_axis: AxType,
  yerr: str,
  overwrite: bool
) -> None:
  plot_df = get_plot_df(df, x_axis)
  output_file = get_output_path(x_axis=x_axis, y_axis=y_axis, plot_gpu=False, overwrite=overwrite)
  cpu_info = get_cpu_info()
  x_label = get_ax_label(x_axis)
  y_label = get_ax_label(y_axis)
  x_col = get_col_name(x_axis)
  y_col = get_col_name(y_axis)
  categories = plot_df[x_col].unique()

  x_label_loc = np.arange(len(categories))
  width = 0.2
  multiplier = 0

  fig, ax = plt.subplots(figsize=(12, 6))

  for alg in plot_df["name"].unique():
    alg_df = plot_df[plot_df["name"] == alg]
    vals = alg_df[y_col].values
    
    if yerr == "ci":
      y_errs = alg_df[f"{y_col}_ci"].values
    elif yerr == "min-max":
      y_errs = (alg_df[f"{y_col}_err_min"].values, alg_df[f"{y_col}_err_max"].values)
    else:
      ValueError(f"Invalid yerr value: {yerr}. Must be 'ci' or 'min-max'.")

    offset = width * multiplier
    ax.bar(x_label_loc + offset, vals, width, label=alg,
           yerr=y_errs, capsize=5)
    multiplier += 1

  # Y-axis things
  ax.set_ylabel(y_label)
  ax.set_ylim(0, max(plot_df[y_col]) * 1.2)
  ax.grid(axis="y", linestyle="--", alpha=1.0)

  # X-axis things
  ax.set_xlabel(x_label)
  ax.set_xticks(x_label_loc + width, labels=categories)

  # General plot things
  ax.set_title(get_plot_title(plot_df, x_axis, cpu_info))
  ax.legend(title="Libraries", loc="upper left", ncols=len(plot_df["name"].unique()))

  plt.tight_layout()
  plt.savefig(output_file, dpi=300)
  plt.close()
  return
  

# def write_gpu_plot(
#   df: pd.DataFrame,
#   x_axis: AxType,
#   y_axis: AxType,
#   overwrite: bool
# ) -> None:
#   # get only the rows where the fixed parameter is set appropriately
#   fixed_col, fixed_val, file_part = get_fixed_param(x_axis)
#   df = df[df[fixed_col] == fixed_val]
#   # sanity_check
#   assert(len(df[fixed_col].unique()) == 1)

#   output_file = get_output_path(x_axis, y_axis, False)
#   cpu_info = get_cpu_info()
#   x_label = get_ax_label(x_axis)
#   y_label = get_ax_label(y_axis)
#   x_col = get_col_name(x_axis)
#   y_col = get_col_name(y_axis)

#   hue_col = "threads_per_gpu_block"
#   unique_hue_vals = df[hue_col].unique()
#   palette = sns.color_palette("tab10", len(unique_hue_vals))

#   sns.set_theme(style="whitegrid")

#   g = sns.FacetGrid(df, col="num_gpu_blocks",
#                     col_wrap=3, aspect=2,
#                     ).set(title=get_plot_title(df, x_axis, cpu_info, True))

  
#   g.map_dataframe(sns.barplot, x=x_col, y=y_col, errorbar=None, hue=hue_col, palette=palette)#.set(yscale="log")
#   g.set_axis_labels(x_label, y_label)
#   g.set_titles(col_template="{col_name} GPU Blocks")
#   g.add_legend(title="Threads per GPU Block")
#   g.legend.set_frame_on(True)  
#   g.despine(left=True)

#   plt.suptitle(get_plot_title(df, x_axis, cpu_info))

#   plt.tight_layout()
#   plt.savefig(output_file, dpi=300)
#   plt.close()
#   return