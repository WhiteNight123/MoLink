#!/usr/bin/env bash
# ===========================================================================
# run_benchmark.sh — Automated benchmark runner for vLLM vs MoLink
#
# What it does:
#   1. Launch two Docker containers (head + tail) on a custom network
#   2. Apply tc/netem network shaping (bandwidth + latency)
#   3. Start either MoLink or vLLM (with Ray)
#   4. Run the Python benchmark client for each RPS value
#   5. Collect JSON results and tear down
#
# Usage:
#   ./run_benchmark.sh               # run all (molink + vllm)
#   ./run_benchmark.sh molink        # only MoLink
#   ./run_benchmark.sh vllm          # only vLLM
#   ./run_benchmark.sh molink vllm   # both, explicit
# ===========================================================================
set -euo pipefail

# =========================== CONFIGURATION ==================================

# Docker
DOCKER_IMAGE="molink:0.1"
DOCKER_NETWORK="molink"
C1_NAME="bench-head"
C2_NAME="bench-tail"
C1_IP="172.26.0.10"
C2_IP="172.26.0.11"
GPU_HEAD=1          # host GPU index for head container
GPU_TAIL=2          # host GPU index for tail container

# Paths inside containers
MODEL_PATH="/gxq/Qwen3-14B"
HOST_TOKENIZER_PATH="/home/emnets-2/gxq/Qwen3-14B"
MOLINK_CODE="/gxq/molink-measurement/MoLink"
MOLINK_PYTHON="/opt/conda/envs/vllm/bin/python"
VLLM_BIN="/opt/conda/envs/vllm/bin/vllm"
RAY_BIN="/opt/conda/envs/vllm/bin/ray"

# Ports
HEAD_HTTP_PORT=8080     # inside container
TAIL_HTTP_PORT=9095
TAIL_HOST_PORT=9095     # mapped to host
HEAD_HOST_PORT=8080     # mapped to host
MOLINK_GRPC_HEAD=50061
MOLINK_GRPC_TAIL=50062
RAY_PORT=6379

# Network conditions to test: "bandwidth,latency"
NETWORK_CONDITIONS=(
    "500mbit,10ms"
    "1gbit,10ms"
    "5gbit,10ms"
    "1gbit,20ms"
    "1gbit,30ms"
)

# Trace parameters
INPUT_TOKENS=1024
OUTPUT_TOKENS=512
RPS_VALUES=(0.5 1 3 5 7)
DURATION=40            # seconds per benchmark run
COOLDOWN=10            # seconds to wait between runs

MAX_MODEL_LEN=4096

# MoLink pipeline config
MAX_CONCURRENT_BATCHES=${MOLINK_MAX_CONCURRENT_BATCHES:-1}

# Results
RESULTS_ROOT="/home/emnets-2/gxq/molink-measurement/MoLink/benchmark/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${RESULTS_ROOT}/${TIMESTAMP}"
BENCHMARK_CLIENT="/home/emnets-2/gxq/molink-measurement/MoLink/benchmark/benchmark_client.py"

# Health check
HEALTH_TIMEOUT=600     # seconds to wait for service startup

# =========================== LOGGING ========================================

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
warn() { log "WARN: $*" >&2; }
die()  { log "ERROR: $*" >&2; exit 1; }

# =========================== CLEANUP ========================================

cleanup() {
    log "Cleaning up containers and processes..."
    docker rm -f "$C1_NAME" "$C2_NAME" 2>/dev/null || true
}
trap cleanup EXIT

# =========================== DOCKER =========================================

ensure_network() {
    if ! docker network inspect "$DOCKER_NETWORK" >/dev/null 2>&1; then
        log "Creating Docker network: $DOCKER_NETWORK"
        docker network create --subnet=172.26.0.0/16 "$DOCKER_NETWORK"
    fi
}

start_containers() {
    log "Starting containers..."
    docker run -d --name "$C1_NAME" \
        -v /mnt/disk1-16/gxq:/data \
        -v /home/emnets-2/gxq:/gxq \
        --gpus "\"device=${GPU_HEAD}\"" \
        --shm-size=64g \
        --cap-add=NET_ADMIN \
        --network "$DOCKER_NETWORK" \
        --ip "$C1_IP" \
        -p "${HEAD_HOST_PORT}:${HEAD_HTTP_PORT}" \
        "$DOCKER_IMAGE" \
        sleep infinity

    docker run -d --name "$C2_NAME" \
        -v /mnt/disk1-16/gxq:/data \
        -v /home/emnets-2/gxq:/gxq \
        --gpus "\"device=${GPU_TAIL}\"" \
        --shm-size=64g \
        --cap-add=NET_ADMIN \
        --network "$DOCKER_NETWORK" \
        --ip "$C2_IP" \
        -p "${TAIL_HOST_PORT}:${TAIL_HTTP_PORT}" \
        "$DOCKER_IMAGE" \
        sleep infinity

    log "Containers started ($C1_NAME=$C1_IP  $C2_NAME=$C2_IP)"
}

# =========================== NETWORK SHAPING ================================

setup_network() {
    local bw="$1"    # e.g. 1gbit
    local delay="$2" # e.g. 10ms

    log "Applying network shaping: ${bw} / ${delay}"

    # Container 1 → Container 2
    docker exec "$C1_NAME" bash -c "\
        tc qdisc del dev eth0 root 2>/dev/null || true; \
        tc qdisc add dev eth0 root handle 1: htb default 30; \
        tc class add dev eth0 parent 1: classid 1:1 htb rate ${bw}; \
        tc qdisc add dev eth0 parent 1:1 netem delay ${delay}; \
        tc filter add dev eth0 parent 1: protocol ip prio 1 u32 \
            match ip dst ${C2_IP} flowid 1:1"

    # Container 2 → Container 1
    docker exec "$C2_NAME" bash -c "\
        tc qdisc del dev eth0 root 2>/dev/null || true; \
        tc qdisc add dev eth0 root handle 1: htb default 30; \
        tc class add dev eth0 parent 1: classid 1:1 htb rate ${bw}; \
        tc qdisc add dev eth0 parent 1:1 netem delay ${delay}; \
        tc filter add dev eth0 parent 1: protocol ip prio 1 u32 \
            match ip dst ${C1_IP} flowid 1:1"

    log "Network shaping applied."
}

# =========================== SERVICE MANAGEMENT =============================

stop_services() {
    log "Stopping services inside containers..."
    # Kill only python/ray processes, not the container's sleep infinity
    docker exec "$C1_NAME" bash -c "pkill -f 'python.*molinkv1' 2>/dev/null || true; pkill -f 'python.*vllm' 2>/dev/null || true; pkill -f 'ray::' 2>/dev/null || true; pkill -f 'raylet' 2>/dev/null || true; pkill -f 'VLLM' 2>/dev/null || true" || true
    docker exec "$C2_NAME" bash -c "pkill -f 'python.*molinkv1' 2>/dev/null || true; pkill -f 'python.*vllm' 2>/dev/null || true; pkill -f 'ray::' 2>/dev/null || true; pkill -f 'raylet' 2>/dev/null || true; pkill -f 'VLLM' 2>/dev/null || true" || true
    sleep 5
}

wait_for_health() {
    local url="${1:-http://localhost:${HEAD_HOST_PORT}/health}"
    local timeout="${2:-$HEALTH_TIMEOUT}"
    local elapsed=0
    log "Waiting for service at ${url} ..."
    while ! curl -sf "$url" >/dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ "$elapsed" -ge "$timeout" ]; then
            # Dump container logs for debugging before failing
            log "=== $C1_NAME logs ==="
            docker exec "$C1_NAME" bash -c "tail -20 /tmp/bench_head.log" 2>/dev/null || true
            log "=== $C2_NAME logs ==="
            docker exec "$C2_NAME" bash -c "tail -20 /tmp/bench_tail.log" 2>/dev/null || true
            die "Health check timed out after ${timeout}s"
        fi
        if [ $((elapsed % 30)) -eq 0 ]; then
            log "  ... still waiting (${elapsed}s/${timeout}s)"
        fi
    done
    log "Service is healthy!"
}

# ---------- MoLink ----------

start_molink() {
    log "Starting MoLink head node (layers 0-20)..."
    docker exec -d -e PYTHONPATH="${MOLINK_CODE}" -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 "$C1_NAME" bash -c "\
        ${MOLINK_PYTHON} -m molinkv1.entrypoints.api_server \
            --model ${MODEL_PATH} \
            --molink-grpc-port ${MOLINK_GRPC_HEAD} \
            --molink-start-layer 0 \
            --molink-end-layer 21 \
            --port ${HEAD_HTTP_PORT} \
            --max-model-len ${MAX_MODEL_LEN} \
            --molink-max-concurrent-batches ${MAX_CONCURRENT_BATCHES} \
            --enforce-eager \
            --no-enable-prefix-caching \
            &>/tmp/bench_head.log"

    # Wait for head to fully start (model loading + gRPC ready)
    wait_for_health "http://localhost:${HEAD_HOST_PORT}/health" 300
    log "Head node is ready. Waiting extra 10s for gRPC stabilization..."
    sleep 10

    log "Starting MoLink tail node (layers 21-end)..."
    docker exec -d -e PYTHONPATH="${MOLINK_CODE}" -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 "$C2_NAME" bash -c "\
        ${MOLINK_PYTHON} -m molinkv1.entrypoints.api_server \
            --model ${MODEL_PATH} \
            --molink-grpc-port ${MOLINK_GRPC_TAIL} \
            --molink-start-layer 21 \
            --molink-end-layer -1 \
            --port ${TAIL_HTTP_PORT} \
            --max-model-len ${MAX_MODEL_LEN} \
            --molink-max-concurrent-batches ${MAX_CONCURRENT_BATCHES} \
            --molink-initial-peer ${C1_IP}:${MOLINK_GRPC_HEAD} \
            --enforce-eager \
            --no-enable-prefix-caching \
            &>/tmp/bench_tail.log"

    # Wait for tail to be ready too
    log "Waiting for tail node health check on port ${TAIL_HOST_PORT}..."
    local elapsed=0
    while ! curl -sf "http://localhost:${TAIL_HOST_PORT}/health" >/dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ "$elapsed" -ge "$HEALTH_TIMEOUT" ]; then
            die "Tail node health check timed out"
        fi
    done
    log "Tail node is ready!"

    # Final wait for gRPC connection to stabilize
    sleep 5
}

# ---------- vLLM + Ray ----------

start_vllm() {
    log "Starting Ray head in container 1..."
    docker exec -d "$C1_NAME" bash -c "\
        CUDA_VISIBLE_DEVICES=0 ${RAY_BIN} start --head \
            --node-ip-address=${C1_IP} \
            --port=${RAY_PORT} --num-gpus=1"

    sleep 8

    log "Starting Ray worker in container 2..."
    docker exec -d "$C2_NAME" bash -c "\
        CUDA_VISIBLE_DEVICES=0 ${RAY_BIN} start \
            --address=${C1_IP}:${RAY_PORT} --num-gpus=1"

    sleep 8

    log "Starting vLLM serve (PP=2)..."
    docker exec -d "$C1_NAME" bash -c "\
        CUDA_VISIBLE_DEVICES=0 ${VLLM_BIN} serve \
            --model ${MODEL_PATH} \
            --port ${HEAD_HTTP_PORT} \
            --max-model-len ${MAX_MODEL_LEN} \
            --pipeline-parallel-size 2 \
            --distributed-executor-backend ray \
            --no-enable-prefix-caching \
            --enforce-eager"

    wait_for_health
}

# =========================== BENCHMARK LOOP =================================

run_benchmarks_for() {
    local system="$1"       # molink | vllm
    local net_label="$2"    # e.g. bw1gbit_delay10ms

    local results_subdir="${RESULTS_DIR}/${system}/${net_label}"
    mkdir -p "$results_subdir"

    for rps in "${RPS_VALUES[@]}"; do
        local outdir="${results_subdir}/rps${rps}"
        mkdir -p "$outdir"

        log "--- [${system}] [${net_label}] rps=${rps} ---"

        if [ "$system" = "molink" ]; then
            python "$BENCHMARK_CLIENT" \
                --url "http://localhost:${HEAD_HOST_PORT}/generate" \
                --type molink \
                --input-tokens "$INPUT_TOKENS" \
                --output-tokens "$OUTPUT_TOKENS" \
                --rps "$rps" \
                --duration "$DURATION" \
                --model "$MODEL_PATH" \
                --tokenizer "$HOST_TOKENIZER_PATH" \
                --output "${outdir}/result.json"
        else
            python "$BENCHMARK_CLIENT" \
                --url "http://localhost:${HEAD_HOST_PORT}/v1/completions" \
                --type vllm \
                --input-tokens "$INPUT_TOKENS" \
                --output-tokens "$OUTPUT_TOKENS" \
                --rps "$rps" \
                --duration "$DURATION" \
                --model "$MODEL_PATH" \
                --tokenizer "$HOST_TOKENIZER_PATH" \
                --output "${outdir}/result.json"
        fi

        log "Done: rps=${rps} → ${outdir}/result.json"
        sleep "$COOLDOWN"
    done
}

# =========================== MAIN ===========================================

main() {
    # Parse which systems to test
    local systems=()
    if [ $# -eq 0 ]; then
        systems=(molink vllm)
    else
        systems=("$@")
    fi

    log "==========================================="
    log " Benchmark suite"
    log " Systems      : ${systems[*]}"
    log " Network conds: ${NETWORK_CONDITIONS[*]}"
    log " RPS values   : ${RPS_VALUES[*]}"
    log " Duration     : ${DURATION}s per run"
    log " Results dir  : ${RESULTS_DIR}"
    log "==========================================="

    mkdir -p "$RESULTS_DIR"

    # Save run metadata
    cat > "${RESULTS_DIR}/config.json" <<EOF
{
  "timestamp": "${TIMESTAMP}",
  "systems": [$(printf '"%s",' "${systems[@]}" | sed 's/,$//')],
  "network_conditions": [$(printf '"%s",' "${NETWORK_CONDITIONS[@]}" | sed 's/,$//')],
  "input_tokens": ${INPUT_TOKENS},
  "output_tokens": ${OUTPUT_TOKENS},
  "rps_values": [$(printf '%s,' "${RPS_VALUES[@]}" | sed 's/,$//')],
  "duration_s": ${DURATION}
}
EOF

    # --- Infrastructure setup ---
    cleanup
    ensure_network
    start_containers

    for system in "${systems[@]}"; do
        for net_cond in "${NETWORK_CONDITIONS[@]}"; do
            bw="${net_cond%%,*}"
            delay="${net_cond##*,}"
            net_label="bw${bw}_delay${delay}"

            log "=========== ${system} | ${net_label} ==========="

            setup_network "$bw" "$delay"
            stop_services

            if [ "$system" = "molink" ]; then
                start_molink
            else
                start_vllm
            fi

            run_benchmarks_for "$system" "$net_label"

            stop_services
        done
    done

    log "==========================================="
    log " All benchmarks complete!"
    log " Results: ${RESULTS_DIR}"
    log "==========================================="
}

main "$@"
