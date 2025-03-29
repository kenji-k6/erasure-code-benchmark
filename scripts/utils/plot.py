from enum import unique
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.hardware_info import CPUInfo, get_cpu_info
from utils.utils import AxType, get_output_path, get_col_name, get_ax_label, get_plot_title
import utils.config as cfg
from typing import Union, Tuple

def get_fixed_param(x_axis: AxType) -> Tuple[str, Union[int, str]]:
  """Returns the fixed parameter for the given AxType."""
  if x_axis == AxType.BLK_SIZE:
    return get_col_name(AxType.FEC_RATIO), cfg.FIXED_FEC_RATIO
  elif x_axis == AxType.FEC_RATIO:
    return get_col_name(AxType.BLK_SIZE), cfg.FIXED_BLOCK_SIZE

def plot_cache_sizes(cpu_info: CPUInfo) -> None:
  pass
  # plt.axvline(x=cpu_info.l1_cache_size_KiB, color="blue", linestyle="--", label="L1 Cache Size")
  # plt.axvline(x=cpu_info.l2_cache_size_KiB, color="red", linestyle="--", label="L2 Cache Size")
  # plt.axvline(x=cpu_info.l3_cache_size_KiB, color="green", linestyle="--", label="L3 Cache Size")

def write_cpu_plot(
  df: pd.DataFrame,
  x_axis: AxType,
  y_axis: AxType,
  cache_sizes: bool
) -> None:
  # get only the rows where the fixed parameter is set appropriately
  fixed_col, fixed_val = get_fixed_param(x_axis)
  df = df[df[fixed_col] == fixed_val]
  # sanity_check
  assert(len(df[fixed_col].unique()) == 1)

  output_file = get_output_path(x_axis, y_axis, False)
  cpu_info = get_cpu_info()
  x_label = get_ax_label(x_axis)
  y_label = get_ax_label(y_axis)
  x_col = get_col_name(x_axis)
  y_col = get_col_name(y_axis)

  sns.set_theme(style="whitegrid",)
  g = sns.catplot(data=df, kind="bar",
              x=x_col, y=y_col, hue="name",
              errorbar=None, palette="tab10",
              aspect=2, legend="full"
  ).set(title=get_plot_title(df, x_axis, cpu_info, False))

  g.ax.set_ylim(0, max(df[y_col]) * 1.2)
  g.set_axis_labels(x_label, y_label) 
  g.despine(left=True)
  g.legend.set_title("Libraries")
  g.legend.set_frame_on(True)
  sns.move_legend(g, "upper right")
  plt.tight_layout()

  plt.savefig(output_file, dpi=300)
  plt.close()
  return

def write_gpu_plot(
  df: pd.DataFrame,
  x_axis: AxType,
  y_axis: AxType,
  cache_sizes: bool
) -> None:
  # get only the rows where the fixed parameter is set appropriately
  fixed_col, fixed_val = get_fixed_param(x_axis)
  if (x_axis == AxType.FEC_RATIO):
    fixed_col = "block_size_B"
    fixed_val = 256*1024
  df = df[df[fixed_col] == fixed_val]
  # sanity_check
  assert(len(df[fixed_col].unique()) == 1)

  print(len(df))
  output_file = get_output_path(x_axis, y_axis, True)
  cpu_info = get_cpu_info()
  x_label = get_ax_label(x_axis)
  y_label = get_ax_label(y_axis)
  x_col = get_col_name(x_axis)
  y_col = get_col_name(y_axis)

  hue_col = "threads_per_gpu_block"
  unique_hue_vals = df[hue_col].unique()
  palette = sns.color_palette("tab10", len(unique_hue_vals))

  sns.set_theme(style="whitegrid")

  g = sns.FacetGrid(df, col="num_gpu_blocks",
                    col_wrap=3, aspect=2,
                    ).set(title=get_plot_title(df, x_axis, cpu_info, True))

  
  g.map_dataframe(sns.barplot, x=x_col, y=y_col, errorbar=None, hue=hue_col, palette=palette)#.set(yscale="log")
  g.set_axis_labels(x_label, y_label)
  g.set_titles(col_template="{col_name} GPU Blocks")
  g.add_legend(title="Threads per GPU Block")
  g.legend.set_frame_on(True)  
  g.despine(left=True)

  plt.suptitle(get_plot_title(df, x_axis, cpu_info, True))

  plt.tight_layout()
  plt.savefig(output_file, dpi=300)
  plt.close()
  return