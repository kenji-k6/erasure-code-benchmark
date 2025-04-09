import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

INPUT_FILE = "results_erasures-500-take-2.csv"
OUTPUT_FILE = "results_erasures-500-take-2.pdf"
Z = 3.291 # 99.9% Confidence interval value

def parse_csv() -> pd.DataFrame:
  df = pd.read_csv(INPUT_FILE)

  # Remove iteration information after the benchmark name
  df["name"] = df["name"].str.split("/", n=1).str[0]

  # Remove runs where more than 32 threads were used
  df = df[df["num_cpu_threads"] <= 32]

  # Ensure no row has an error message
  assert df["err_msg"].isna().all(), "Some rows have an error message"

  # Sort dataframe by benchmark name
  df.sort_values(by=["name", "num_cpu_threads"], ascending=[False, True], inplace=True)

  # Compute the block size in KiB
  df["block_size_KiB"] = df["block_size_B"] // 1024

  # Compute the encode throughput (mean, min, max) in Gbps
  df["throughput_Gbps"] = (df["message_size_B"] * 8) / df["time_ns"] # equal to (#bits / 10^9) / (t_ns / 10^9) = #Gbits / s
  df["throughput_Gbps_stddev"] = df["throughput_Gbps"] * (df["time_ns_stddev"] / df["time_ns"]) # first order taylor expansion/error propagation
  df["throughput_Gbps_err_min"] = df["throughput_Gbps"]-((df["message_size_B"] * 8) / df["time_ns_max"])
  df["throughput_Gbps_err_max"] = ((df["message_size_B"] * 8) / df["time_ns_min"])-df["throughput_Gbps"]

  df["throughput_Gbps_ci"] = Z * (df["throughput_Gbps_stddev"] / np.sqrt(df["num_iterations"]))


  # Assert that the block size and FEC params are the same for each row
  assert df["block_size_KiB"].nunique() == 1, "Block size is not the same for all rows"
  assert df["FEC"].nunique() == 1, "FEC params are not the same for all rows"
  return df


def main() -> None:
  # Read the CSV file, compute the throughput and adjust unit for the block size
  df = parse_csv()

  # Variables for the plot (will be visualized)
  # Change as needed
  x_label = "Num. CPU Threads"
  y_label = "Encode Throughput (Gbps)"
  title = "MDS vx XOR libraries"

  # Variables for the plot (will be used to generate the plot)
  # Do not change
  x_col = "num_cpu_threads"
  y_col = "throughput_Gbps"

  # Line below can either be confidence interval or min-max
  # y_err_low_col, y_err_up_col = f"{y_col}_err_min", f"{y_col}_err_max" 
  y_err_low_col, y_err_up_col = f"{y_col}_ci", f"{y_col}_ci"
  algorithms = df["name"].unique()

  fec_values = df[x_col].unique()
  x_label_loc = np.arange(len(fec_values)) # locations of the x-axis labels

  width = 0.2 # width of the bars
  multiplier = 0 # multiplier for the x-axis locations
  total_width = width * len(algorithms) # total width of the bars per CPU thread group

  # Set font size to 15
  plt.rcParams.update({'font.size': 15})

  # Create the figure and axis
  fig, ax = plt.subplots(figsize=(7,5))

  # Set the X-axis labels & ticks
  ax.set_xlabel(x_label)
  ax.set_xticks((x_label_loc-width/2)+total_width/2, labels=fec_values)

  # Set the Y-axis label scale and grid
  # ax.set_yscale("log")
  ax.set_ylabel(y_label)
  ax.grid(axis="y", linestyle="--", alpha=1.0, which="both")



  # Plot each algorithm individually
  for alg in algorithms:
    alg_df = df[df["name"] == alg]
    vals = alg_df[y_col].values
    y_errs = (alg_df[y_err_low_col].values, alg_df[y_err_up_col].values)

    # Compute the offset for the current algorithm
    offset = width * multiplier

    # Plot the bars
    ax.bar(x_label_loc+offset,
           height=vals, width=width, label=alg
           )
    # Plot the error bars (min/max)
    ax.errorbar(
        x_label_loc+offset,
        y=vals,
        yerr=y_errs,
        fmt="none",
        ecolor="black",
        capsize=5,
        capthick=1,
        alpha=0.5
    )
    
    multiplier += 1

  # Set the title & legend
  ax.set_title(title)
  ax.legend(loc="upper left", ncols=1)

  plt.tight_layout()
  plt.savefig(OUTPUT_FILE, format="pdf", dpi=300)
  plt.close()


if __name__ == "__main__":
  main()