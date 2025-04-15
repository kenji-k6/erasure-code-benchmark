import os
import utils.data as data
import utils.plot as plot

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
RAW_DIR = os.path.join(SCRIPT_DIR, "../results/raw")
PLOT_DIR = os.path.join(SCRIPT_DIR, "../results/plots")

CPU_COMPARISON_FILE = os.path.join(RAW_DIR, "cpu_ec_results_old.csv")
XOREC_COMPARISON_FILE = os.path.join(RAW_DIR, "xorec_ec_results.csv")
XOREC_GPU_COMPARISON_FILE = os.path.join(RAW_DIR, "xorec_gpu_ec_results.csv")


def ensure_directories() -> None:
  if not os.path.exists(RAW_DIR):
    os.makedirs(RAW_DIR)
  if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)





if __name__ == "__main__":
  ensure_directories()
  
  df = data.get_df(CPU_COMPARISON_FILE)

  plot.plot_EC(df, "encode", PLOT_DIR)
  
