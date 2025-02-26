import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# File / directory paths
INPUT_FILE = './results/raw/benchmark_results_test.csv'
OUTPUT_DIR = './results/processed/'



def plot_xbuffer_size__y_time(df: pd.DataFrame):
  # Set plot style
  sns.set_theme(style="whitegrid")

  # Create the plot
  plt.figure(figsize=(10,6))

  sns.scatterplot(data=df, x='tot_bytes', y='real_time', hue='name', palette='tab10')

  # Customize the plot
  plt.title('Benchmark Results', fontsize=16)
  plt.xlabel('Total Buffer Size (bytes)', fontsize=12)
  plt.ylabel('Real Time (seconds)', fontsize=12)
  plt.xscale('log')  # If you want to use log scale for x-axis
  plt.yscale('log')  # If you want to use log scale for y-axis
  plt.legend(title='Benchmark Names', bbox_to_anchor=(1.05, 1), loc='upper left')

  # Save the plot to a file
  plt.tight_layout()
  plt.savefig('./results/processed/stored_results.png')

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
  df['real_time'] = df['real_time'] / 1e9
  df['cpu_time'] = df['cpu_time'] / 1e9

  plot_xbuffer_size__y_time(df) 