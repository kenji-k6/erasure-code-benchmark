
# Erasure Code Benchmark
A high-performance benchmarking suite for evaluation open-source erasure coding (EC)
libraries on **x86_64 machines**. This repository provides a structured micro-benchmarking testbed
based on [Google's microbenchmark support library](https://github.com/google/benchmark)
to analyze various EC implementations based on multiple system parameters.


## Features
* **Multi-Library Support**: Benchmarks [cm256](https://github.com/catid/cm256),
[Intel ISA-L](https://github.com/intel/isa-l), [Leopard](https://github.com/catid/leopard)
and [Wirehair](https://github.com/catid/wirehair) as well as a custom baseline implementation.

* **Customizable Parameters**: Evaluate performance based on payload size, parity size,
algorithm and SIMD (amongst others).

* **GPU Memory Support**: The baseline XOR-EC algorithm also supports GPU operations. See the documentation in the xorec/ directory.

* **Micro-Benchmarking Suite**: Optimized benchmarking framework in C++.

* **Plotting**: Provides a pre-implemented Python script to visualize your results.

* **Benchmarking Results**: You can also find a full result set of all EC libraries that was
ran on [Swis National Supercomputing Centre's](https://www.cscs.ch/) nodes.


## Building
### Prerequisites
* **Build System**: CMake 3.16+

* **Make**: GNU 'make'

* **C++ Compiler**: GCC 10+ (Recommended) / Clang 12+ (must support C++20)

* **Python**: 3.8+ (for result analysis and visualization)

* **SIMD Support**: AVX2 / AVX-512 recommended for best performance

* **NVIDIA CUDA Toolkit** CUDA 11.4.0+

### Building the Benchmark framework (Linux)
1. **Clone the repository and update/initialize Git submodules**
```bash
git clone https://github.com/kenji-k6/erasure-code-benchmark.git
cd erasure-code-benchmark
git submodule update --init --recursive
```

2. **Install required packages**
```bash
sudo apt update && sudo apt install -y \
  build-essential libbenchmark-dev automake \
  autoconf libtool libomp-dev
```

3. **Build ISA-L (has to be built manually)**
```bash
cd libraries/isa-l
make -f Makefile.unx
cd ../..
```


4. **Create build folder & compile benchmark program**
```bash
mkdir build && cd build
cmake ..
make
```


### Building the the Result Analysis
1. **Create and activate a Python virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

2. **Install required Python packages**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Run the plotting script by running python3 scripts/main.py**
