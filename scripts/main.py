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

  # OPEN_SOURCE category
  dir = os.path.join(PLOT_DIR, CATEGORY_INFO[Category.OPEN_SOURCE][OUTPUT_DIR])
  plot.plot_EC(
    data=df.copy(),
    y_type=PlotType.ENCODE,
    category=Category.OPEN_SOURCE,
    output_dir=dir
  )
  plot.plot_blocksize(
    data=df.copy(),
    y_type=PlotType.ENCODE,
    category=Category.OPEN_SOURCE,
    output_dir=dir
  )
  plot.plot_cpu_threads(
    data=df.copy(),
    y_type=PlotType.ENCODE,
    category=Category.OPEN_SOURCE,
    output_dir=dir
  )
  plot.plot_lost_blocks(
    data=df.copy(),
    y_type=PlotType.DECODE,
    category=Category.OPEN_SOURCE,
    output_dir=dir
  )

  # SIMD plots
  dir = os.path.join(PLOT_DIR, CATEGORY_INFO[Category.SIMD][OUTPUT_DIR])
  plot.plot_EC(
    data=df.copy(),
    y_type=PlotType.ENCODE,
    category=Category.SIMD,
    output_dir=dir
  )
  plot.plot_blocksize(
    data=df.copy(),
    y_type=PlotType.ENCODE,
    category=Category.SIMD,
    output_dir=dir
  )
  plot.plot_cpu_threads(
    data=df.copy(),
    y_type=PlotType.ENCODE,
    category=Category.SIMD,
    output_dir=dir
  )

  # XOR-EC plots
  dir = os.path.join(PLOT_DIR, CATEGORY_INFO[Category.XOREC][OUTPUT_DIR])
  plot.plot_EC(
    data=df.copy(),
    y_type=PlotType.ENCODE,
    category=Category.XOREC,
    output_dir=dir
  )
  plot.plot_blocksize(
    data=df.copy(),
    y_type=PlotType.ENCODE,
    category=Category.XOREC,
    output_dir=dir
  )
  plot.plot_lost_blocks(
    data=df.copy(),
    y_type=PlotType.DECODE,
    category=Category.XOREC,
    output_dir=dir
  )

  # XOREC_GPU plots
  dir = os.path.join(PLOT_DIR, CATEGORY_INFO[Category.XOREC_GPU][OUTPUT_DIR])
  plot.plot_EC(
    data=df.copy(),
    y_type=PlotType.ENCODE,
    category=Category.XOREC_GPU,
    output_dir=dir
  )

  # Paper comparison plots
  dir = os.path.join(PLOT_DIR, CATEGORY_INFO[Category.PAPER_COMPARISON][OUTPUT_DIR])
  plot.plot_paper_comparison(
    data=df.copy(),
    data_paper=paper_df.copy(),
    y_type=PlotType.ENCODE,
    category=Category.PAPER_COMPARISON,
    output_dir=dir
  )

  # Small SIMD plots
  dir = os.path.join(PLOT_DIR, CATEGORY_INFO[Category.SMALL_SIMD][OUTPUT_DIR])
  plot.plot_EC(
    data=df.copy(),
    y_type=PlotType.ENCODE,
    category=Category.SMALL_SIMD,
    output_dir=dir
  )
  plot.plot_blocksize(
    data=df.copy(),
    y_type=PlotType.ENCODE,
    category=Category.SMALL_SIMD,
    output_dir=dir
  )

  # Recoverability probability plots
  plot.plot_P_recoverable(output_dir=os.path.join(PLOT_DIR, "misc"))

  
