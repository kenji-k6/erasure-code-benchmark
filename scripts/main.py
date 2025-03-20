import argparse
import os
from utils.compute import get_ec_dfs, get_perf_dfs
import utils.config as cfg
from utils.plot import write_plot
from utils.utils import get_benchmark_names, get_axes, ensure_paths, get_plot_id




def main() -> None:
  parser = argparse.ArgumentParser(description="Plot benchmark results")
  parser.add_argument(
    "--input-dir",
    type=str,
    help="input directory name (in /results/raw/)",
    required=False
  )

  parser.add_argument(
    "--output-dir",
    type=str,
    help="optional output directory for the plots (created inside /results/plots/)",
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
    help="x-axis parameter"
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
    help="y-axis parameter(s)"
  )

  parser.add_argument(
    "--y-scale",
    type=str,
    choices=["linear", "log"],
    default="log",
    help="y-axis scale"
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
    help="list of benchmarks to plot"
  )

  parser.add_argument(
    "--confidence-intervals",
    action="store_true",
    help="include confidence intervals in the plots"
  )

  parser.add_argument(
    "--cache-sizes",
    action="store_true",
    help="include cache sizes in the plots"
  )

  parser.add_argument(
    "--theoretical-bounds",
    action="store_true",
    help="include theoretical bounds (Scalar, AVX, AVX2, AVX512) in the plots"
  )

  args = parser.parse_args()

  if args.input_dir:
    cfg.INPUT_DIR = args.input_dir

  cfg.EC_INPUT_FILE = os.path.join(cfg.RAW_DIR, cfg.INPUT_DIR, cfg.EC_FILE_NAME)
  cfg.PERF_INPUT_FILE = os.path.join(cfg.RAW_DIR, cfg.INPUT_DIR, cfg.PERF_FILE_NAME)

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


  ensure_paths(check_perf_file=theoretical_bounds)
  ec_dfs = get_ec_dfs(cfg.EC_INPUT_FILE, selected_benchmarks)
  perf_dfs = get_perf_dfs(cfg.PERF_INPUT_FILE) if theoretical_bounds else None

  for x_axis in x_axes:
    plot_id = get_plot_id(x_axis)
    for y_axis in y_axes:
      write_plot(
        ec_df=ec_dfs[plot_id],
        perf_dfs=perf_dfs,
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