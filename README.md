
# ECC Benchmark

## Toolchain & Installation

1. Install C++ compiler and bechmarking dependencies:
`sudo apt update `
`sudo apt install -y build-essential cmake git clang libbenchmark-dev automake autoconf libtool build-essential python3 python3.12-venv`
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


