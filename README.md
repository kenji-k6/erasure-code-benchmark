
# ECC Benchmark

## Toolchain & Installation

1. Install C++ compiler and bechmarking dependencies:
`sudo apt update `
`sudo apt install -y build-essential cmake git clang libbenchmark-dev automake autoconf libtool build-essential`
`sudo apt-get install libom-dev`

2. Install Python tools (for results analysis)
`pip install numpy pandas matplotlib`

3. Update git submodules
`git submodule update --init --recursive`

4. ISA-L needs to be built manually. In root-folder execute the following:
`cd libraries/isa-l`
`./autogen.sh`
`./configure`
`make`
`cd ../..`

5. Create the build folder, use cmake, build the benchmark program
`mkdir build`
`cd build`
`cmake ..`
`make`
`./ecc-benchmark`


## Explanation of current directory tree:

```bash
ecc-benchmark/                     # Root directory
│
│── include/                       # Header files directory
│   │── benchmark_runner.h         
│   │── benchmark.h                # Header for the base ECCBenchmark class (abstract interface)
│   │── cm256_benchmark.h
│   │── leopard_benchmark.h
│   │── utils.h                    # Header for utility functions and macros
│   │── wirehair_benchmark.h
│
│── libraries/                     # Directory for third-party libraries
│   │── aff3ct/                    # unused as of now (TODO)
│   │── cm256/                     
│   │── isa-l/                     # unused as of now (TODO)
│   │── leopard/
│   │── wirehair/                  # currently working on this
│
│── results/                       # Directory for storing benchmark results
│   │── processed/                 # Processed results (e.g., aggregated data, graphs) (TODO)
│   │── raw/                       # Raw results (e.g., CSV files, logs) (TODO)
│
│── scripts/                       # Directory for scripts (e.g., data processing, plotting) (TODO)
│
│── src/                           # Directory for source files (.cpp)
│   │── library_benchmark/         # Directory for library-specific benchmark implementations
│   │   │── cm256_benchmark.cpp    # Implementation of CM256Benchmark class
│   │   │── leopard_benchmark.cpp  # Implementation of LeopardBenchmark class
│   │   │── wirehair_benchmark.cpp # Implementation of WirehairBenchmark class
│   │── benchmark_runner.cpp       # Implementation of BenchmarkRunner class (manages benchmarks)
│   │── main.cpp                   # Main entry point for the application
│   │── utils.cpp                  # Implementation of utility functions
│
│── CMakeLists.txt                 # CMake build configuration file
│── README.md                      # Project documentation (overview, setup instructions)
│── requirements.txt               # Python dependencies (unused as of now, will only be needed for result processing) (TODO)

```

