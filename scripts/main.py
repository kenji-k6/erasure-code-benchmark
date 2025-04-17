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
  
  if not os.path.exists(os.path.join(PLOT_DIR, OUTPUT_SUBDIR[Category.CPU])):
    os.makedirs(os.path.join(PLOT_DIR, OUTPUT_SUBDIR[Category.CPU]))
  if not os.path.exists(os.path.join(PLOT_DIR, OUTPUT_SUBDIR[Category.GPU])):
    os.makedirs(os.path.join(PLOT_DIR, OUTPUT_SUBDIR[Category.GPU]))
  if not os.path.exists(os.path.join(PLOT_DIR, OUTPUT_SUBDIR[Category.SIMD])):
    os.makedirs(os.path.join(PLOT_DIR, OUTPUT_SUBDIR[Category.SIMD]))
  if not os.path.exists(os.path.join(PLOT_DIR, OUTPUT_SUBDIR[Category.XOREC])):
    os.makedirs(os.path.join(PLOT_DIR, OUTPUT_SUBDIR[Category.XOREC]))




if __name__ == "__main__":
  ensure_directories()

  df = data.get_df(RAW_FILE)

  # CPU comparisons
  plot.plot_EC(
    df,
    y_type=PlotType.ENCODE,
    category=Category.CPU,
    output_dir=os.path.join(PLOT_DIR, OUTPUT_SUBDIR[Category.CPU])
  )
  plot.plot_datasize(
    df,
    y_type=PlotType.ENCODE,
    category=Category.CPU,
    output_dir=os.path.join(PLOT_DIR, OUTPUT_SUBDIR[Category.CPU])
  )
  plot.plot_ec_datasize_heatmap(
    df,
    y_type=PlotType.ENCODE,
    category=Category.CPU,
    output_dir=os.path.join(PLOT_DIR, OUTPUT_SUBDIR[Category.CPU])
  )


  # SIMD comparisons
  plot.plot_EC(
    df,
    y_type=PlotType.ENCODE,
    category=Category.SIMD,
    output_dir=os.path.join(PLOT_DIR, OUTPUT_SUBDIR[Category.SIMD])
  )
  plot.plot_datasize(
    df,
    y_type=PlotType.ENCODE,
    category=Category.SIMD,
    output_dir=os.path.join(PLOT_DIR, OUTPUT_SUBDIR[Category.SIMD])
  )
  
