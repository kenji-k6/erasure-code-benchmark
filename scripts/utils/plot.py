import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from utils.config import AVX_BITS, AVX_XOR_CPI, AVX_COLOR, AVX2_BITS, AVX2_XOR_CPI, AVX2_COLOR
from utils.hardware_info import CPUInfo, get_cpu_info
from utils.theoretical_bounds import get_theoretical_bound_func
from utils.utils import AxType, get_output_path, get_col_name, get_ax_label, get_plot_id, get_plot_title

def plot_cache_sizes(cpu_info: CPUInfo) -> None:
  """Plot the cache sizes on the graph."""
  plt.axvline(x=cpu_info.l2_cache_size_KiB, color="red", linestyle="--", label="L2 Cache Size")
  plt.axvline(x=cpu_info.l3_cache_size_KiB, color="green", linestyle="--", label="L3 Cache Size")


def plot_xticks(df: pd.DataFrame, x_axis: AxType) -> None:
  """Plot the x-axis ticks."""
  ax = plt.gca()
  x_col = get_col_name(x_axis)
  x_ticks = sorted(df[x_col].unique())

  if x_axis == AxType.BUF_SIZE:
    x_ticklabels = [(lambda x: f"{x//1024} MiB" if x >= 1024 else f"{x} KiB")(x) for x in x_ticks]
    rot = 45
  else:
    x_ticklabels = [f"{x}" for x in x_ticks]
    rot = 0

  ax.set_xticks(ticks=x_ticks, labels=x_ticklabels, rotation=rot)
  ax.set_xlim(left=x_ticks[0]/2, right=x_ticks[-1]*2)


def plot_yticks() -> None:
  """Plot the y-axis' (minor) ticks."""
  ax = plt.gca()
  ax.tick_params(
    axis="y",
    which="minor",
    left=True
  )
  ax.yaxis.set_minor_formatter(ticker.LogFormatter())
  ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))


def plot_confidence_intervals(df: pd.DataFrame, x_axis: AxType, y_axis: AxType) -> None:
  """Plot the confidence intervals."""
  x_col = get_col_name(x_axis)
  y_col = get_col_name(y_axis)
  low_err_col = f"{y_col}_lower"
  up_err_col = f"{y_col}_upper"
  ax = plt.gca()

  for name, group in df.groupby("name"):
    plt.errorbar(
      group[x_col],
      group[y_col],
      yerr=[group[low_err_col].to_numpy(), group[up_err_col].to_numpy()],
      fmt='o',
      color=ax.get_legend().legend_handles[
        df["name"]
        .unique()
        .tolist()
        .index(name)
      ].get_color(),
      capsize=5
    )


def plot_avx_avx2_xor(df: pd.DataFrame, x_axis: AxType, y_axis: AxType, cpu_info: CPUInfo) -> None:
  x_col = get_col_name(x_axis)
  func = get_theoretical_bound_func(y_axis)

  params_df = df[[
    "block_size_B",
    "num_data_blocks",
    "num_parity_blocks",
    "num_lost_blocks"
  ]].drop_duplicates().sort_values(by=x_col, ascending=True)

  x_values = params_df[x_col].unique()

  y_values_AVX = params_df.apply(
    lambda row: func(
      AVX_XOR_CPI,
      AVX_BITS,
      cpu_info.clock_speed_GHz,
      row["block_size_B"],
      row["num_data_blocks"],
      row["num_parity_blocks"],
      row["num_lost_blocks"]
    ),
    axis=1
  ).tolist()

  y_values_AVX2 = params_df.apply(
    lambda row: func(
      AVX2_XOR_CPI,
      AVX2_BITS,
      cpu_info.clock_speed_GHz,
      row["block_size_B"],
      row["num_data_blocks"],
      row["num_parity_blocks"],
      row["num_lost_blocks"]
    ),
    axis=1
  ).tolist()

  assert(len(x_values) == len(y_values_AVX) == len(y_values_AVX2))
  plt.plot(x_values, y_values_AVX, label="AVX XOR", color=AVX_COLOR, linestyle="--")
  plt.plot(x_values, y_values_AVX2, label="AVX2 XOR", color=AVX2_COLOR, linestyle="--")



def write_plot(
  df: pd.DataFrame,
  x_axis: AxType,
  y_axis: AxType,
  y_scale: str,
  confidence_intervals: bool=True,
  cache_sizes: bool=True,
  theoretical_bounds: bool=True
) -> None:
  plot_id = get_plot_id(x_axis)
  output_file = get_output_path(x_axis, y_axis, y_scale)
  cpu_info = get_cpu_info()
  x_label = get_ax_label(x_axis)
  y_label = get_ax_label(y_axis)
  x_col = get_col_name(x_axis)
  y_col = get_col_name(y_axis)

  sns.set_theme(style="whitegrid")
  plt.figure(figsize=(10, 6))
  sns.scatterplot(data=df, x=x_col, y=y_col, hue="name", palette="tab10")

  plt.xlabel(x_label, fontsize=12)
  plt.ylabel(y_label, fontsize=12)

  plt.xscale("log", base=2)
  plt.yscale(y_scale)

  if cache_sizes and x_axis == AxType.BUF_SIZE: plot_cache_sizes(cpu_info)
  if theoretical_bounds: plot_avx_avx2_xor(df, x_axis, y_axis, cpu_info)
  if confidence_intervals: plot_confidence_intervals(df, x_axis, y_axis)

  plt.legend(
    title="Libraries",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
  )
  plt.title(get_plot_title(df, cpu_info), fontsize=12)

  plot_xticks(df, x_axis)
  if y_scale == "log": plot_yticks()

  plt.tight_layout()
  plt.savefig(output_file, dpi=300)
  plt.close()
  return