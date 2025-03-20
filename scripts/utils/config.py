import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'results')
RAW_DIR = os.path.join(RESULTS_DIR, 'raw')
PLOT_DIR = os.path.join(RESULTS_DIR, 'plots')
EC_FILE_NAME = "ec_results.csv"
PERF_FILE_NAME = "perf_results.csv"
INPUT_DIR = ""
OUTPUT_DIR = None
EC_INPUT_FILE = None
PERF_INPUT_FILE = None


# Z value used for plotting the confidence intervals
# 80% => 1.282, 85% => 1.440, 90% => 1.645, 95% => 1.960,
# 99% => 2.576, 99.9% => 3.291
Z_VALUE = 3.291