import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# File / directory paths
INPUT_FILE = './results/raw/benchmark_results_test.csv'
OUTPUT_DIR = './results/processed/'



def plot_x_buffer_size__y_time(df: pd.DataFrame):
  file_name = 'buffer_size_vs_time.png'

  # Set plot style
  sns.set_theme(style="whitegrid")

  # Create the plot
  plt.figure(figsize=(10,6))

  sns.scatterplot(data=df, x='total_MB', y='real_time_ms', hue='name', palette='tab10')

  # Customize the plot
  plt.title('Benchmark Results', fontsize=16)
  plt.xlabel('Total Buffer Size (MB)', fontsize=12)
  plt.ylabel('Real Time (milliseconds)', fontsize=12)
  plt.xscale('log')  # If you want to use log scale for x-axis
  plt.yscale('log')  # If you want to use log scale for y-axis
  plt.legend(title='Benchmark Names', bbox_to_anchor=(1.05, 1), loc='upper left')

  # Save the plot to a file
  plt.tight_layout()
  plt.savefig(OUTPUT_DIR+file_name)

  # Show the plot
  plt.show()
  

def plot_x_buffer_size__y_throughout(df: pd.DataFrame):
  file_name = 'buffer_size_vs_throughput.png'

  # Set plot style
  sns.set_theme(style="whitegrid")

  # Create the plot
  plt.figure(figsize=(10,6))

  sns.scatterplot(data=df, x='total_MB', y='throughput_Gbps', hue='name', palette='tab10')

  # Customize the plot
  plt.title('Benchmark Results', fontsize=16)
  plt.xlabel('Total Buffer Size (MB)', fontsize=12)
  plt.ylabel('Throughput (Gbps)', fontsize=12)
  plt.xscale('log')  # If you want to use log scale for x-axis
  plt.yscale('log')  # If you want to use log scale for y-axis
  plt.legend(title='Benchmark Names', bbox_to_anchor=(1.05, 1), loc='upper left')

  # Save the plot to a file
  plt.tight_layout()
  plt.savefig(OUTPUT_DIR+file_name)

  # Show the plot
  plt.show()


if __name__ == "__main__":
  # Make sure output directory exists
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  df = pd.read_csv(INPUT_FILE)

  # Remove "BM_" from the start and "/iterations:..." from the end of the 'name' column
  df['name'] = df['name'].str.replace(r'^BM_|/iterations:.*$', '', regex=True)

  # Remove rows where corruption occured
  df = df[df['err_msg'].isna()]

  # Convert nanoseconds to seconds
  df['real_time_ms'] = df['real_time'] / 1e6
  df['cpu_time_ms'] = df['cpu_time'] / 1e6

  # compute the throughput
  df['total_MB'] = df['tot_bytes'] / 1e6
  df['throughput_Gbps'] = (df['total_MB'] * 8) / (df['real_time_ms'] * 1e3)

  plot_x_buffer_size__y_time(df)
  plot_x_buffer_size__y_throughout(df)