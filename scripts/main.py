import argparse
import os
from utils.compute import get_df
import utils.config as cfg
from utils.plot import write_cpu_plot, write_gpu_plot
from utils.utils import get_axis, ensure_paths




def main() -> None:
  parser = argparse.ArgumentParser(description="Plot benchmark results")
  parser.add_argument(
    "--input",
    type=str,
    help="input file name (in /results/raw/)",
    required=True
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
      "block-size",
      "fec",
      "gpu-params"
    ],
    default="fec",
    help="x-axis parameter"
  )

  parser.add_argument(
    "--y-axis",
    type=str,
    choices=[
      "encode-time",
      "encode-throughput"
    ],
    default="encode-throughput",
    help="y-axis parameter"
  )

  parser.add_argument(
    "--fixed-block-size",
    type=int,
    choices=[
      4,
      8,
      16,
      32,
      64,
      128,
      256
    ],
    default=4,
    help="fixed block size in KiB"
  )

  parser.add_argument(
    "--fixed-fec",
    type=str,
    choices=[
      "2:1",
      "4:2",
      "8:4",
      "16:4",
      "16:8",
      "32:4",
      "32:8"
    ],
    default="32:8",
    help="fixed fec ratio"
  )

  parser.add_argument(
    "--cache-sizes",
    action="store_true",
    help="include cache sizes in the plots"
  )

  parser.add_argument(
    "--gpu",
    action="store_true",
    help="plot gpu results"
  )

  args = parser.parse_args()

  if args.input:
    cfg.INPUT_FILE = os.path.join(cfg.RAW_DIR, args.input)

  if args.output_dir:
    cfg.OUTPUT_DIR = os.path.join(cfg.PLOT_DIR, args.output_dir)
  else:
    cfg.OUTPUT_DIR = cfg.PLOT_DIR

  cfg.FIXED_BLOCK_SIZE = args.fixed_block_size

  fec_x, fec_y = args.fixed_fec.split(":")

  cfg.FIXED_FEC_RATIO = f"FEC({int(fec_x)},{int(fec_y)})"

  
  x_axis = get_axis(args.x_axis)
  y_axis = get_axis(args.y_axis)

  cache_sizes = args.cache_sizes
  plot_gpu = args.gpu


  ensure_paths()
  df = get_df(cfg.INPUT_FILE, plot_gpu)
  if plot_gpu:
    write_gpu_plot(df=df, x_axis=x_axis, y_axis=y_axis, cache_sizes=cache_sizes)
  else:
    write_cpu_plot(df=df, x_axis=x_axis, y_axis=y_axis, cache_sizes=cache_sizes)

  return

if __name__ == "__main__":
  main()