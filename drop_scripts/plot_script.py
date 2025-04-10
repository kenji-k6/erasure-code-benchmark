import os
import re
import matplotlib.pyplot as plt
import numpy as np
BASE_DIR = "drop-rate-test/logs-long-run"


def parse_experiment_dir_name(dirname) -> dict[str, int | bool]:
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
  for dir in dirs:
    if dir.startswith(temp_dir_name):
      return dir



def main():
  
  for dirname in os.listdir(BASE_DIR):
    dirpath = os.path.join(BASE_DIR, dirname)
    if os.path.isdir(dirpath):
      temp = parse_experiment_dir_name(dirname)
      get_dir_name(temp["proc"], temp["buf"], temp["fq"], temp["fq_num"], temp["trial"])
      

if __name__ == "__main__":
  main()
  # Further processing and plotting can be added here