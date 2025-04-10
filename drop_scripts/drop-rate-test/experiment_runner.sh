#!/bin/bash

# This script executes one trial of a single experiment configuration.
#
# Usage:
#   ./experiment_runner.sh <server_ip> <server_nic_ip> <experiment_root_directory> <client_processes> <buffer_size> <time> <trial_number> [fq_rate]
#
#   server_ip                - Remote server IP address.
#   server_nic_ip            - The NIC IP on the server to use by iperf3 on the client.
#   experiment_root_directory- Base directory where experiment results will be stored.
#   client_processes         - Number of iperf3 client processes (-P option).
#   buffer_size              - Buffer size (MTU) for iperf3 (-l option).
#   time                     - Time (in seconds) for the iperf3 test (-t option).
#   trial_number             - A sequential trial number for this configuration.
#   fq_rate (optional)       - If provided (and not "nofq"), the iperf3 client will include --fq-rate=<fq_rate>.

if [ "$#" -lt 7 ] || [ "$#" -gt 8 ]; then
    echo "Usage: $0 <server_ip> <server_nic_ip> <experiment_root_directory> <client_processes> <buffer_size> <time> <trial_number> [fq_rate]"
    exit 1
fi

SERVER_IP="$1"
SERVER_NIC_IP="$2"
EXPERIMENT_ROOT="$3"
CLIENT_PROCESSES="$4"
BUF_SIZE="$5"
TIME="$6"
TRIAL_NUMBER="$7"

# Process the optional fq_rate parameter.
if [ "$#" -eq 8 ]; then
    FQ_RATE="$8"
    OPTIONAL_FQRATE="--fq-rate=${FQ_RATE}"
    FQ_DIR_TAG="fq-${FQ_RATE}"
else
    OPTIONAL_FQRATE=""
    FQ_DIR_TAG="nofq"
fi

# Create a unique directory for this trial.
EXPERIMENT_DIR="${EXPERIMENT_ROOT}/experiment-proc${CLIENT_PROCESSES}-buf${BUF_SIZE}-time${TIME}-${FQ_DIR_TAG}-trial${TRIAL_NUMBER}-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$EXPERIMENT_DIR"

echo "Starting trial ${TRIAL_NUMBER} for configuration: processes=${CLIENT_PROCESSES}, buffer_size=${BUF_SIZE}, time=${TIME}, fq_rate=${FQ_DIR_TAG}"
echo "Logs will be saved in $EXPERIMENT_DIR"

# Clean previous temporary files on the server.
ssh "$SERVER_IP" "rm -f /tmp/server_snmp.log /tmp/iperf3_server.log /tmp/iperf3_server.pid"

# Log initial UDP SNMP counters on the server.
ssh "$SERVER_IP" "echo '=== Initial UDP SNMP counters ===' > /tmp/server_snmp.log; grep -i Udp /proc/net/snmp >> /tmp/server_snmp.log"

# Start the iperf3 server on the remote host.
ssh "$SERVER_IP" "nohup numactl -C 0-15 iperf3 -s >> /tmp/iperf3_server.log 2>&1 & echo \$! > /tmp/iperf3_server.pid"

# Give the server a moment to start.
sleep 2

echo "Running iperf3 client test for trial ${TRIAL_NUMBER}..."
numactl -C 0-15 iperf3 -u -c "$SERVER_NIC_IP" -l "$BUF_SIZE" -b 100G -t "$TIME" -P "$CLIENT_PROCESSES" $OPTIONAL_FQRATE | tee "${EXPERIMENT_DIR}/client_iperf.log"

# Log final UDP SNMP counters on the server.
ssh "$SERVER_IP" "echo '=== Final UDP SNMP counters ===' >> /tmp/server_snmp.log; grep -i Udp /proc/net/snmp >> /tmp/server_snmp.log"

# Shut down the iperf3 server.
ssh "$SERVER_IP" "kill -2 \$(cat /tmp/iperf3_server.pid) && rm -f /tmp/iperf3_server.pid"

# Retrieve the full server logs.
scp "$SERVER_IP:/tmp/server_snmp.log" "$EXPERIMENT_DIR/"

# Fetch and save the last 100 lines of the iperf3 server log.
ssh "$SERVER_IP" "tail -n 100 /tmp/iperf3_server.log" > "${EXPERIMENT_DIR}/iperf3_server_tail.log"

echo "Trial ${TRIAL_NUMBER} completed, logs saved in $EXPERIMENT_DIR"
