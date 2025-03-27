import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from utils.hardware_info import CPUInfo, get_cpu_info
from utils.utils import AxType, get_output_path, get_col_name, get_ax_label, get_plot_title


def plot_cache_sizes(cpu_info: CPUInfo) -> None:
  pass
  # plt.axvline(x=cpu_info.l1_cache_size_KiB, color="blue", linestyle="--", label="L1 Cache Size")
  # plt.axvline(x=cpu_info.l2_cache_size_KiB, color="red", linestyle="--", label="L2 Cache Size")
  # plt.axvline(x=cpu_info.l3_cache_size_KiB, color="green", linestyle="--", label="L3 Cache Size")

def write_plot(
  df: pd.DataFrame,
  x_axis: AxType,
  y_axis: AxType,
  cache_sizes: bool
) -> None:
  output_file = get_output_path(x_axis, y_axis)
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
  ).set(title=get_plot_title(df, x_axis, cpu_info))

  g.ax.set_ylim(0, max(df[y_col]) * 1.2)
  g.set_axis_labels(x_label, y_label) 
  g.despine(left=True) # Remove left spine

  g.legend.set_title("Libraries")
  g.legend.set_frame_on(True)
  sns.move_legend(g, "upper right")
  plt.tight_layout()
  # g.label(get_plot_title(df, x_axis, cpu_info))
  


  # if cache_sizes and x_axis == AxType.BLK_SIZE: plot_cache_sizes(cpu_info)

  plt.savefig(output_file, dpi=300)
  plt.close()
  return