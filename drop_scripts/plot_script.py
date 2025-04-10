import os
import re
import matplotlib.pyplot as plt
import numpy as np
from typing import TypedDict, Tuple
BASE_DIR = "drop-rate-test/logs-long-run"

class ExperimentConfig(TypedDict):
  proc: int
  buf: int
  fq: bool
  fq_num: int
  trial: int


def parse_experiment_dir_name(dirname) -> ExperimentConfig:
  """
  Parses the directory name to extract configuration parameters.
  """
  pattern = r"proc(\d+)-buf(\d+).*?(nofq|fq)(?:-(\d+)G)?-trial(\d+)"
  match = re.search(pattern, dirname)

  if match:
    return {
      "proc": int(match.group(1)),
      "buf": int(match.group(2)),
      "fq": match.group(3) == "fq",
      "fq_num": int(match.group(4)) if match.group(4) else -1,
      "trial": int(match.group(5))
    }
  else:
    raise ValueError(f"Invalid directory name format: {dirname}")

def get_dir_name(proc: int, buf: int, fq: bool, fq_num: int, trial: int) -> str:
  proc_str = f"proc{proc}"
  buf_str = f"buf{buf}"
  fq_str = f"fq-{fq_num}G" if fq else "nofq"
  trial_str = f"trial{trial}"

  dirs = os.listdir(BASE_DIR)
  temp_dir_name = f"experiment-{proc_str}-{buf_str}-time15-{fq_str}-{trial_str}-"
  res = []
  for dir in dirs:
    if dir.startswith(temp_dir_name):
      res.append(dir)
  
  assert len(res) == 1, f"Expected one directory, found {len(res)} for {temp_dir_name}"
  return res[0]




def parse_client_iperf_log(dirname: str) -> Tuple[int, int]:
  """
  Parses the client_iperf.log file to extract dropped and total packets from the receiver line.
  """
  filepath = os.path.join(BASE_DIR, dirname, "client_iperf.log")
  with open(filepath, "r") as f:
    lines = f.readlines()

  pattern = r"(\d+)/(\d+)\s+\(\d+(\.\d+)?%\)"
  for line in reversed(lines): #reverse for faster parsing
    if "receiver" in line:
      match = re.search(pattern, line)

      if match:
        dropped = int(match.group(1))
        total = int(match.group(2))
        return dropped, total
      
      raise ValueError(f"Invalid line format in client_iperf.log: {line}")
  raise ValueError("No receiver line found in client_iperf.log")
      
  




def main():
  
  for dirname in os.listdir(BASE_DIR):
    dirpath = os.path.join(BASE_DIR, dirname)
    if os.path.isdir(dirpath):
      experiment_config = parse_experiment_dir_name(dirname)
      dropped, total = parse_client_iperf_log(dirname)
      


      

if __name__ == "__main__":
  main()
  # Further processing and plotting can be added here