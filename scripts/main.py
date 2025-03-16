import argparse
import os
import pandas as pd
from utils.compute import get_grouped_dataframes
import utils.config as cfg
from utils.plot import write_plot
from utils.utils import AxType, get_benchmark_names, get_axes, ensure_paths, get_plot_id




def main() -> None:
  parser = argparse.ArgumentParser(description="Plot benchmark results")
  parser.add_argument(
    "--input-file",
    type=str,
    help="Input file name (in /results/raw/)",
    required=True
  )

  parser.add_argument(
    "--output-dir",
    type=str,
    help="Optional output directory for the plots (created inside /results/plots/)",
    required=False
  )

  parser.add_argument(
    "--x-axis",
    type=str,
    choices=[
      "buffer-size",
      "num-parity-blocks",
      "num-lost-blocks"
    ],
    default="buffer-size",
    help="X-axis parameter"
  )

  parser.add_argument(
    "--y-axis",
    type=str,
    choices=[
      "encode-time",
      "decode-time",
      "encode-throughput",
      "decode-throughput"
    ],
    nargs="+",
    default=["encode-time", "decode-time", "encode-throughput", "decode-throughput"],
    help="Y-axis parameter(s)"
  )

  parser.add_argument(
    "--y-scale",
    type=str,
    choices=["linear", "log"],
    default="log",
    help="Y-axis scale"
  )

  parser.add_argument(
    "--selected-benchmarks",
    type=str,
    nargs="+",
    choices=[
      "cm256",
      "isal",
      "leopard",
      "wirehair",
      "xorec"
    ],
    default=[],
    help="List of benchmarks to plot"
  )

  parser.add_argument(
    "--confidence-intervals",
    action="store_true",
    help="Include confidence intervals in the plots"
  )

  parser.add_argument(
    "--cache-sizes",
    action="store_true",
    help="Include cache sizes in the plots"
  )

  parser.add_argument(
    "--theoretical-bounds",
    action="store_true",
    help="Include theoretical bounds (AVX, AVX2) in the plots"
  )

  args = parser.parse_args()

  cfg.INPUT_FILE = os.path.join(cfg.RAW_DIR, args.input_file)

  if args.output_dir:
    cfg.OUTPUT_DIR = os.path.join(cfg.PLOT_DIR, args.output_dir)
  else:
    cfg.OUTPUT_DIR = cfg.PLOT_DIR
  
  x_axes = get_axes([args.x_axis])

  y_axes = get_axes(args.y_axis)

  y_scale = args.y_scale

  selected_benchmarks = get_benchmark_names(args.selected_benchmarks)

  confidence_intervals = args.confidence_intervals
  cache_sizes = args.cache_sizes
  theoretical_bounds = args.theoretical_bounds


  ensure_paths()
  dfs = get_grouped_dataframes(cfg.INPUT_FILE, selected_benchmarks)

  for x_axis in x_axes:
    plot_id = get_plot_id(x_axis)
    for y_axis in y_axes:
      write_plot(
        df=dfs[plot_id],
        x_axis=x_axis,
        y_axis=y_axis,
        y_scale=y_scale,
        confidence_intervals=confidence_intervals,
        cache_sizes=cache_sizes,
        theoretical_bounds=theoretical_bounds
      )
  return


if __name__ == "__main__":
  main()