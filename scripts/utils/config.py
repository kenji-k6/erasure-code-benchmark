import os

# Input/Output directories and file constants
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'results')
RAW_DIR = os.path.join(RESULTS_DIR, 'raw')
PLOT_DIR = os.path.join(RESULTS_DIR, 'plots')
INPUT_FILE = None

# AVX(2) related constants
AVX_XOR_CPI_XEON = 0.33
AVX_COLOR = "cornflowerblue"
AVX_BITS = 128

AVX2_XOR_CPI_XEON = 0.33
AVX2_BITS = 256
AVX2_COLOR = "indigo"