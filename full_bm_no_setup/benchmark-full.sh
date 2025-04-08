#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
OUTPUT_FILE=""
ITERATIONS=""
SIMD_SELECTIONS=()
CMAKE_SIMD_FLAGS=""
BM_SIMD_FLAGS=""


# Function to print usage
usage() {
  echo "Usage: $0 -o|--output <output_file> -i|--iterations <num_iterations> --simd [avx|avx2|avx512]"
  exit 1
}

# Parse flags using getopts
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    -i|--iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --simd)
      IFS=',' read -ra SIMD_SELECTIONS <<< "$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      usage
      ;;
  esac
done

# Check if params are missing
if [ -z "$OUTPUT_FILE" ] || [ -z "$ITERATIONS" ]; then
  echo "Error: Missing required arguments."
  usage
fi

if [ ${#SIMD_SELECTIONS[@]} -eq 0 ]; then
  echo "Error: No SIMD selections provided."
  usage 1
fi


# Validate iterations is a number and >= 3
if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]]; then
  echo "Error: --iterations must be a positive integer."
  usage 1
elif [[ "$ITERATIONS" -lt 3 ]]; then
  echo "Error: --iterations must be at least 3."
  usage 1
fi

# Validate output file has .csv extension
if [[ "$OUTPUT_FILE" != *.csv ]]; then
  echo "Error: --output must be a CSV file (ending with .csv)."
  usage 1
fi

for simd in "${SIMD_SELECTIONS[@]}"; do
  case "$simd" in
    avx)
      CMAKE_SIMD_FLAGS+=" -mavx"
      BM_SIMD_FLAGS+=",xorec-avx"
      ;;
    avx2)
      CMAKE_SIMD_FLAGS+=" -mavx2"
      BM_SIMD_FLAGS+=",xorec-avx2"
      ;;
    avx512)
      CMAKE_SIMD_FLAGS+=" -mavx512f"
      BM_SIMD_FLAGS+=",xorec-avx512"
      ;;
    *)
      echo "Error: Invalid SIMD selection '$simd'. Valid options are: avx, avx2, avx512."
      usage 1
      ;;
  esac
done

# Function to print progress
report_progress() {
  local input="$*"
  local green="\033[1;32m"
  local reset="\033[0m"

  printf "${green}%s${reset}\n" "$input"
}


# Create temporary directory for benchmarking
WORK_DIR=$(mktemp -d bench-XXXX)
report_progress "Setting environment variables"
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export GOMP_CPU_AFFINITY="0-59"
export CC=$(which gcc)
export CXX=$(which g++)
echo "OMP_PROC_BIND=$OMP_PROC_BIND, OMP_PLACES=$OMP_PLACES, GOMP_CPU_AFFINITY=$GOMP_CPU_AFFINITY"
echo "CC=$CC"
echo "CXX=$CXX"

err_cleanup() {
  local input="$*"
  local red="\033[1;31m"
  local reset="\033[0m"

  printf "${red}%s${reset}\n" "$input"
  exit 1
}


report_progress "Extracting benchmarking files to temporary directory $WORK_DIR"
tar -xzf $SCRIPT_DIR/erasure-code-benchmark-full.tar.gz -C $WORK_DIR --strip-components=1

BUILD_DIR=$(mkdir -p "$WORK_DIR/build" && realpath "$WORK_DIR/build")
RESULT_DIR=$(mkdir -p "$WORK_DIR/results/raw" && realpath "$WORK_DIR/results/raw")
ISAL_DIR=$(realpath "$WORK_DIR/libraries/isa-l")


cd "$ISAL_DIR"
report_progress "Building ISA-L"
make -f Makefile.unx || { err_cleanup "ISA-L build failed"; }
report_progress "Built ISA-L"

cd "$BUILD_DIR"
report_progress "Running CMake configuration"
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="$CMAKE_SIMD_FLAGS $CMAKE_CXX_FLAGS" .. || { err_cleanup "CMake configuration failed"; }
report_progress "CMake configuration completed"

report_progress "Building benchmark"
make > /dev/null || { err_cleanup "Benchmark build failed"; }
report_progress "Built benchmark"

report_progress "Starting benchmark execution"
./ec-benchmark --iterations "$ITERATIONS" -c "isal$BM_SIMD_FLAGS" || { err_cleanup "Benchmark execution failed"; }

report_progress "Copying results to $OUTPUT_FILE"
cd "$SCRIPT_DIR"
cp "$RESULT_DIR"/results.csv "$OUTPUT_FILE" || { err_cleanup "Result saving failed"; }
report_progress "Cleaning up temporary files..."
rm -rf "$WORK_DIR"

report_progress "Done. Results saved to $OUTPUT_FILE"
exit 0