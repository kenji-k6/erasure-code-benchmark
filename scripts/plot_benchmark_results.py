import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# File / directory paths
INPUT_FILE = './results/raw/benchmark_results.csv'
OUTPUT_DIR = './results/processed/'


##### VARYING BUFFER SIZE PLOTS #####
def bufferSize_vs_time(df: pd.DataFrame):
  file_name = 'buffer_size_vs_time.png'

  # Get the constant parameters to mention in plot title
  num_data_blocks = df.iloc[0]['tot_data_size_B'] // df.iloc[0]['block_size_B']
  redundancy_ratio = df.iloc[0]['redundancy_ratio']
  num_lost_blocks = df.iloc[0]['num_lost_blocks']
  num_iterations = df.iloc[0]['num_iterations']
  title = f"#Data Blocks: {num_data_blocks}, #Redundancy Ratio: {redundancy_ratio}, #Lost Blocks: {num_lost_blocks}, #Iterations: {num_iterations}"


  # Get the measured buffer sizes
  buffer_sizes = sorted(df['tot_data_size_MiB'].unique())

  # Set plot style
  sns.set_theme(style="whitegrid")

  # Create the plot
  plt.figure(figsize=(10,6))
  
  sns.scatterplot(data=df, x='tot_data_size_MiB', y='time_ms', hue='name', palette='tab10')

  # Customize the plot
  plt.title(title, fontsize=12)
  plt.xlabel("Buffer Size (MiB)", fontsize=12)
  plt.ylabel("Time (ms)", fontsize=12)
  plt.xscale('log')
  plt.yscale('linear')
  plt.legend(title='Libraries', bbox_to_anchor=(1.05, 1), loc='upper left')

  # Properly set x-ticks
  ax = plt.gca()
  ax.set_xticks(buffer_sizes)
  ax.set_xticklabels([str(sz) + " MiB" for sz in buffer_sizes])

  # Save the plot to a file
  plt.tight_layout()
  plt.savefig(OUTPUT_DIR+file_name)



def bufferSize_vs_throughput(df: pd.DataFrame):
  file_name = 'buffer_size_vs_throughput.png'

  # Get the constant parameters to mention in plot title
  num_data_blocks = df.iloc[0]['tot_data_size_B'] // df.iloc[0]['block_size_B']
  redundancy_ratio = df.iloc[0]['redundancy_ratio']
  num_lost_blocks = df.iloc[0]['num_lost_blocks']
  num_iterations = df.iloc[0]['num_iterations']
  title = f"#Data Blocks: {num_data_blocks}, #Redundancy Ratio: {redundancy_ratio}, #Lost Blocks: {num_lost_blocks}, #Iterations: {num_iterations}"


  # Get the measured buffer sizes
  var_buffer_sizes = sorted(df['tot_data_size_MiB'].unique())

  # Set plot style
  sns.set_theme(style="whitegrid")

  # Create the plot
  plt.figure(figsize=(10,6))
  
  sns.scatterplot(data=df, x='tot_data_size_MiB', y='throughput_Gbps', hue='name', palette='tab10')

  # Customize the plot
  plt.title(title, fontsize=12)
  plt.xlabel("Buffer Size (MiB)", fontsize=12)
  plt.ylabel("Throughput (Gbps)", fontsize=12)
  plt.xscale('log')
  plt.yscale('linear')
  plt.legend(title='Libraries', bbox_to_anchor=(1.05, 1), loc='upper left')

  # Properly set x-ticks
  ax = plt.gca()
  ax.set_xticks(var_buffer_sizes)
  ax.set_xticklabels([str(sz) + " MiB" for sz in var_buffer_sizes])

  # Save the plot to a file
  plt.tight_layout()
  plt.savefig(OUTPUT_DIR+file_name)
##### VARYING BUFFER SIZE PLOTS #####


##### FIXED BUFFER SIZE PLOTS #####
def redundancyRatio_vs_time(df: pd.DataFrame):
  file_name = 'redundancy_ratio_vs_time.png'

  # Get the constant parameters to mention in plot title
  buffer_size_mib = df.iloc[0]['tot_data_size_MiB']
  num_data_blocks = df.iloc[0]['num_data_blocks']
  num_lost_blocks = df.iloc[0]['num_lost_blocks']
  num_iterations = df.iloc[0]['num_iterations']
  title = f"Buffer Size: {buffer_size_mib} MiB, #Data Blocks: {num_data_blocks}, #Lost Blocks: {num_lost_blocks}, #Iterations: {num_iterations}"

  recovery_blocks_ticks = sorted(df['num_recovery_blocks'].unique()); 

  # Set plot style
  sns.set_theme(style="whitegrid")

  # Create the plot
  plt.figure(figsize=(10,6))
  
  sns.scatterplot(data=df, x='num_recovery_blocks', y='time_ms', hue='name', palette='tab10')

  # Customize the plot
  plt.title(title, fontsize=12)
  plt.xlabel("#Parity Blocks", fontsize=12)
  plt.ylabel("Time (ms)", fontsize=12)
  plt.xscale('log')
  plt.yscale('linear')
  plt.legend(title='Libraries', bbox_to_anchor=(1.05, 1), loc='upper left')

  # Properly set x-ticks
  ax = plt.gca()
  ax.set_xticks(recovery_blocks_ticks)
  ax.set_xticklabels([str(sz) for sz in recovery_blocks_ticks])

  # Save the plot to a file
  plt.tight_layout()
  plt.savefig(OUTPUT_DIR+file_name)







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

  # Helper Columns
  df['tot_data_size_MiB'] = df['tot_data_size_B'] // (1024 * 1024)
  df['throughput_Gbps'] = (df['tot_data_size_B']*8) / (df['time_ns'])
  df['num_data_blocks'] = df['tot_data_size_B'] // df['block_size_B']
  df['num_recovery_blocks'] = (df['redundancy_ratio'] * df['num_data_blocks']).astype(int)

  # Group Dataframe by plot_id
  dfs = {plot_id: group for plot_id, group in df.groupby('plot_id')}

  # print(dfs[0].head())
  # print(dfs[1].head())
  # print(dfs[2].head())

  bufferSize_vs_time(dfs[0])
  bufferSize_vs_throughput(dfs[0])
  
  redundancyRatio_vs_time(dfs[1])

  # # compute the throughput
  # df['total_MB'] = df['tot_bytes'] / 1e6
  # df['throughput_Gbps'] = (df['total_MB'] * 8) / (df['real_time_ms'] * 1e3)

  # plot_x_buffer_size__y_time(df)
  # plot_x_buffer_size__y_throughout(df)