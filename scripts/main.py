import os
import utils.data as data
import utils.plot as plot
from utils.utils import Category, PlotType, OUTPUT_SUBDIR

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
RAW_FILE = os.path.join(SCRIPT_DIR, "../results/raw", "ec_results.csv")
PLOT_DIR = os.path.join(SCRIPT_DIR, "../results/plots")


def ensure_directories() -> None:
  if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

  for category in Category:
    category_dir = os.path.join(PLOT_DIR, OUTPUT_SUBDIR[category])
    if not os.path.exists(category_dir):
      os.makedirs(category_dir)
  
  misc_dir = os.path.join(PLOT_DIR, "misc")
  if not os.path.exists(misc_dir):
    os.makedirs(misc_dir)




if __name__ == "__main__":
  ensure_directories()
  df = data.get_df(RAW_FILE)

  # Plot the bar plots for encoding and decoding throughput for each category
  for category in [Category.CPU, Category.SIMD, Category.XOREC, Category.OPENSOURCE]:
    category_dir = os.path.join(PLOT_DIR, OUTPUT_SUBDIR[category])
    for y_type in PlotType:
      plot.plot_EC(
        data=df.copy(),
        y_type=y_type,
        category=category,
        output_dir=category_dir
      )
      plot.plot_datasize(
        data=df.copy(),
        y_type=y_type,
        category=category,
        output_dir=category_dir
      )
    plot.plot_lost_blocks(
      data=df.copy(),
      y_type=PlotType.DECODE,
      category=category,
      output_dir=category_dir
    )
  

  # Plot the heatmap for encoding throughput for CPU category
  plot.plot_ec_datasize_heatmap(
    data=df.copy(),
    val_type=PlotType.ENCODE,
    category=Category.CPU,
    output_dir=os.path.join(PLOT_DIR, OUTPUT_SUBDIR[Category.CPU])
  )

  plot.plot_P_recoverable(output_dir=os.path.join(PLOT_DIR, "misc"))

  
