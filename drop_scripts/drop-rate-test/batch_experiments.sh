#!/bin/bash
#
# Batch experiment runner.
#
# This script executes batches of experiments by iterating first over a number of trials
# (outer loop) and then over combinations of MTU sizes, client process counts, and fq-rate options.
#
# Usage:
#   ./batch_experiments.sh <server_ip> <server_nic_ip> <experiment_root_directory> <time> <trial_count>
#
#   server_ip                 - Remote server IP address.
#   server_nic_ip             - The NIC IP on the server to use by iperf3 on the client.
#   experiment_root_directory - Base directory where experiment results will be stored.
#   time                      - Time (in seconds) for the iperf3 test.
#   trial_count               - Total number of trials.
#

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <server_ip> <server_nic_ip> <experiment_root_directory> <time> <trial_count>"
    exit 1
fi

SERVER_IP="$1"
SERVER_NIC_IP="$2"
EXPERIMENT_ROOT="$3"
TIME="$4"
TRIAL_COUNT="$5"

# Define the arrays of configuration parameters.
mtu_sizes=(1024 2048 4096 8192)       # used as buffer_size (-l)
proc_nums=(1 2 4 8 16)                  # used as number of iperf3 client processes (-P)
fq_rates=(1G 5G 10G nofq)             # if "nofq", the iperf3 fq-rate option is omitted

# Path to the single experiment runner script.
EXPERIMENT_SCRIPT="./experiment_runner.sh"

# Outer loop: iterate over the trial number.
for trial in $(seq 1 "$TRIAL_COUNT"); do
    echo "====================================================="
    echo "Starting trial ${trial} out of ${TRIAL_COUNT}"
    echo "====================================================="
    # Inner loops over all configuration combinations.
    for mtu in "${mtu_sizes[@]}"; do
        for proc in "${proc_nums[@]}"; do
            for fq in "${fq_rates[@]}"; do
                echo "-----------------------------------------------------"
                echo "Running configuration: MTU=${mtu}, Processes=${proc}, Time=${TIME}, fq_rate=${fq}, Trial=${trial}"
                echo "-----------------------------------------------------"
                # Call the experiment runner with the current trial number.
                if [ "$fq" = "nofq" ]; then
                    "$EXPERIMENT_SCRIPT" "$SERVER_IP" "$SERVER_NIC_IP" "$EXPERIMENT_ROOT" "$proc" "$mtu" "$TIME" "$trial"
                else
                    "$EXPERIMENT_SCRIPT" "$SERVER_IP" "$SERVER_NIC_IP" "$EXPERIMENT_ROOT" "$proc" "$mtu" "$TIME" "$trial" "$fq"
                fi
                # Optional: Insert a pause between each experiment configuration if needed.
                sleep 2
            done
        done
    done
done

echo "All batch experiments completed."
