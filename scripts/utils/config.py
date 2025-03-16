import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'results')
RAW_DIR = os.path.join(RESULTS_DIR, 'raw')
PLOT_DIR = os.path.join(RESULTS_DIR, 'plots')
OUTPUT_DIR = None
INPUT_FILE = None


AVX_BITS = 128
AVX_XOR_CPI = 0.33
AVX_COLOR = "cornflowerblue"

AVX2_BITS = 256
AVX2_XOR_CPI = 0.33
AVX2_COLOR = "indigo"

# Z value used for plotting the confidence intervals
# 80% => 1.282, 85% => 1.440, 90% => 1.645, 95% => 1.960,
# 99% => 2.576, 99.9% => 3.291
Z_VALUE = 3.291