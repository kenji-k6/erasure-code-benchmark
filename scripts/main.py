import os
import utils.data as data
import utils.plot as plot
from utils.utils import Category, PlotType, CATEGORY_INFO, OUTPUT_DIR

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
RAW_FILE = os.path.join(SCRIPT_DIR, "../results/raw", "final_results.csv")
PAPER_RAW_FILE = os.path.join(SCRIPT_DIR, "../results/raw", "paper_final_results.csv")
PLOT_DIR = os.path.join(SCRIPT_DIR, "../results/plots")


def ensure_directories() -> None:
  if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

  for category in Category:
    category_dir = os.path.join(PLOT_DIR, CATEGORY_INFO[category][OUTPUT_DIR])
    if not os.path.exists(category_dir):
      os.makedirs(category_dir)
  
  misc_dir = os.path.join(PLOT_DIR, "misc")
  if not os.path.exists(misc_dir):
    os.makedirs(misc_dir)




if __name__ == "__main__":
  ensure_directories()
  df = data.get_df(RAW_FILE)
  paper_df = data.get_df(PAPER_RAW_FILE)

  # Plot the bar plots for encoding and decoding throughput for each category
  for category in [Category.OPEN_SOURCE, Category.SIMD, Category.XOREC]:
    category_dir = os.path.join(PLOT_DIR, CATEGORY_INFO[category][OUTPUT_DIR])
    plot.plot_EC(
      data=df.copy(),
      y_type=PlotType.ENCODE,
      category=category,
      output_dir=category_dir
    )
    plot.plot_blocksize(
      data=df.copy(),
      y_type=PlotType.ENCODE,
      category=category,
      output_dir=category_dir
    )


  plot.plot_paper_comparison(
    data=df.copy(),
    data_paper=paper_df.copy(),
    y_type=PlotType.ENCODE,
    category=Category.PAPER_COMPARISON,
    output_dir=os.path.join(PLOT_DIR, CATEGORY_INFO[Category.PAPER_COMPARISON][OUTPUT_DIR])
  )


  plot.plot_P_recoverable(output_dir=os.path.join(PLOT_DIR, "misc"))

  
