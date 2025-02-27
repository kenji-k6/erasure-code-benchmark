import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# File / directory paths
INPUT_FILE = './results/raw/benchmark_results.csv'
OUTPUT_DIR = './results/processed/'

  

# def plot_x_buffer_size__y_throughout(df: pd.DataFrame):
#   file_name = 'buffer_size_vs_throughput.png'

#   # Set plot style
#   sns.set_theme(style="whitegrid")

#   # Create the plot
#   plt.figure(figsize=(10,6))

#   sns.scatterplot(data=df, x='total_MB', y='throughput_Gbps', hue='name', palette='tab10')

#   # Customize the plot
#   plt.title('Benchmark Results', fontsize=16)
#   plt.xlabel('Total Buffer Size (MB)', fontsize=12)
#   plt.ylabel('Throughput (Gbps)', fontsize=12)
#   plt.xscale('log')  # If you want to use log scale for x-axis
#   plt.yscale('log')  # If you want to use log scale for y-axis
#   plt.legend(title='Benchmark Names', bbox_to_anchor=(1.05, 1), loc='upper left')

#   # Save the plot to a file
#   plt.tight_layout()
#   plt.savefig(OUTPUT_DIR+file_name)

#   # Show the plot
#   plt.show()

def bufferSize_vs_time(df: pd.DataFrame):
  file_name = 'buffer_size_vs_time.png'

  # Get buffer size in MB
  df['tot_data_size_MB'] = df['tot_data_size_B'] / 1e6

  # Set plot style
  sns.set_theme(style="whitegrid")

  # Create the plot
  plt.figure(figsize=(10,6))
  
  sns.scatterplot(data=df, x='tot_data_size_MB', y='time_ms', hue='name', palette='tab10')

  # Customize the plot
  plt.figtext(0.5, -0.05, "This is a small additional information about the plot", ha="center", va="top", fontsize=12)
  plt.xlabel("Buffer Size (MB)", fontsize=12)
  plt.ylabel("Time (ms)", fontsize=12)
  plt.xscale('log')
  plt.yscale('log')
  plt.legend(title='Libraries', bbox_to_anchor=(1.05, 1), loc='upper left')

  # Save the plot to a file
  plt.tight_layout()
  plt.savefig(OUTPUT_DIR+file_name)

  plt.show()

  






if __name__ == "__main__":
  # Make sure output directory exists
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  df = pd.read_csv(INPUT_FILE)

  # Remove "/iterations:..." from the end of the 'name' column
  df['name'] = df['name'].str.split('/', n=1).str[0]

  # Remove rows where corruption occured
  df = df[df['err_msg'].isna()]

  # Convert nanoseconds to milliseconds
  df['time_ms'] = df['time_ns'] / 1e6

  # Group Dataframe by plot_id
  dfs = {plot_id: group for plot_id, group in df.groupby('plot_id')}

  # print(dfs[0].head())
  # print(dfs[1].head())
  # print(dfs[2].head())

  bufferSize_vs_time(dfs[0])
  

  # # compute the throughput
  # df['total_MB'] = df['tot_bytes'] / 1e6
  # df['throughput_Gbps'] = (df['total_MB'] * 8) / (df['real_time_ms'] * 1e3)

  # plot_x_buffer_size__y_time(df)
  # plot_x_buffer_size__y_throughout(df)