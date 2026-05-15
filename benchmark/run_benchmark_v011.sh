#!/usr/bin/env bash
# ===========================================================================
# run_benchmark_v011.sh — Compare MoLink v0.11 vs v0.19
#
# v0.11: conda env "molink" (vllm 0.11.2), code in /home/MoLink
# v0.19: conda env "vllm"  (vllm 0.19.0), code in /gxq/molink-measurement/MoLink
#
# Usage:
#   ./run_benchmark_v011.sh              # run both
#   ./run_benchmark_v011.sh molink011    # only v0.11
#   ./run_benchmark_v011.sh molink019    # only v0.19
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
GPU_HEAD=1
GPU_TAIL=2

# Paths
MODEL_PATH="/gxq/Qwen3-14B"
HOST_TOKENIZER_PATH="/home/emnets-2/gxq/Qwen3-14B"

# v0.11 (container code, molink conda env)
V011_PYTHON="/opt/conda/envs/molink/bin/python"
V011_PYTHONPATH="/home/MoLink"

# v0.19 (external code, vllm conda env)
V019_PYTHON="/opt/conda/envs/vllm/bin/python"
V019_PYTHONPATH="/gxq/molink-measurement/MoLink"

# Ports
HEAD_HTTP_PORT=8080
HEAD_HOST_PORT=8080
TAIL_HTTP_PORT=9095
TAIL_HOST_PORT=9095
MOLINK_GRPC_HEAD=50061
MOLINK_GRPC_TAIL=50062

# Network conditions
NETWORK_CONDITIONS=(
    "1gbit,10ms"
)

# Trace parameters
INPUT_TOKENS=1024
OUTPUT_TOKENS=512
RPS_VALUES=(3)
DURATION=30
COOLDOWN=10
MAX_MODEL_LEN=4096

# Results
RESULTS_ROOT="/home/emnets-2/gxq/molink-measurement/MoLink/benchmark/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${RESULTS_ROOT}/${TIMESTAMP}"
BENCHMARK_CLIENT="/home/emnets-2/gxq/molink-measurement/MoLink/benchmark/benchmark_client.py"

HEALTH_TIMEOUT=300

# =========================== LOGGING ========================================

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { log "ERROR: $*" >&2; exit 1; }

# =========================== CLEANUP ========================================

cleanup() {
    log "Cleaning up containers..."
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
    sleep 5
}

# =========================== NETWORK SHAPING ================================

setup_network() {
    local bw="$1" delay="$2"
    log "Applying network shaping: ${bw} / ${delay}"

    docker exec "$C1_NAME" bash -c "\
        tc qdisc del dev eth0 root 2>/dev/null || true; \
        tc qdisc add dev eth0 root handle 1: htb default 30; \
        tc class add dev eth0 parent 1: classid 1:1 htb rate ${bw}; \
        tc qdisc add dev eth0 parent 1:1 netem delay ${delay}; \
        tc filter add dev eth0 parent 1: protocol ip prio 1 u32 \
            match ip dst ${C2_IP} flowid 1:1"

    docker exec "$C2_NAME" bash -c "\
        tc qdisc del dev eth0 root 2>/dev/null || true; \
        tc qdisc add dev eth0 root handle 1: htb default 30; \
        tc class add dev eth0 parent 1: classid 1:1 htb rate ${bw}; \
        tc qdisc add dev eth0 parent 1:1 netem delay ${delay}; \
        tc filter add dev eth0 parent 1: protocol ip prio 1 u32 \
            match ip dst ${C1_IP} flowid 1:1"

    log "Network shaping applied."
}

reset_network() {
    log "Removing network shaping..."
    docker exec "$C1_NAME" bash -c "tc qdisc del dev eth0 root 2>/dev/null || true"
    docker exec "$C2_NAME" bash -c "tc qdisc del dev eth0 root 2>/dev/null || true"
}

# =========================== SERVICE MANAGEMENT =============================

_pkill_services() {
    docker exec "$1" bash -c "\
        pkill -f 'python.*molinkv1' 2>/dev/null || true; \
        pkill -f 'python.*vllm' 2>/dev/null || true; \
        pkill -f 'VLLM' 2>/dev/null || true" || true
}

stop_services() {
    log "Stopping services inside containers..."
    _pkill_services "$C1_NAME"
    _pkill_services "$C2_NAME"
    sleep 5
}

wait_for_health() {
    local url="${1:-http://localhost:${HEAD_HOST_PORT}/health}"
    local timeout="${2:-$HEALTH_TIMEOUT}"
    local elapsed=0
    log "Waiting for service at ${url} ..."
    while ! curl -sf --noproxy localhost "$url" >/dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ "$elapsed" -ge "$timeout" ]; then
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

# =========================== MOLINK v0.11 ===================================
# Container code at /home/MoLink, molink conda env (vllm 0.11.2)

start_molink_v011() {
    log "=== MoLink v0.11 head node (layers 0-21) ==="
    docker exec -d -e PYTHONPATH="${V011_PYTHONPATH}" \
        -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 \
        "$C1_NAME" bash -c "\
        ${V011_PYTHON} -m molinkv1.entrypoints.api_server \
            --model ${MODEL_PATH} \
            --molink-grpc-port ${MOLINK_GRPC_HEAD} \
            --molink-start-layer 0 \
            --molink-end-layer 21 \
            --port ${HEAD_HTTP_PORT} \
            --max-model-len ${MAX_MODEL_LEN} \
            --molink-enabled \
            --enforce-eager \
            --no-enable-prefix-caching \
            &>/tmp/bench_head.log"

    wait_for_health "http://localhost:${HEAD_HOST_PORT}/health" 300
    log "Head node ready. Waiting 20s for gRPC stabilization..."
    sleep 20

    log "=== MoLink v0.11 tail node (layers 21-end) ==="
    docker exec -d -e PYTHONPATH="${V011_PYTHONPATH}" \
        -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 \
        "$C2_NAME" bash -c "\
        ${V011_PYTHON} -m molinkv1.entrypoints.api_server \
            --model ${MODEL_PATH} \
            --molink-grpc-port ${MOLINK_GRPC_TAIL} \
            --molink-start-layer 21 \
            --molink-end-layer -1 \
            --port ${TAIL_HTTP_PORT} \
            --max-model-len ${MAX_MODEL_LEN} \
            --molink-initial-peer ${C1_IP}:${MOLINK_GRPC_HEAD} \
            --molink-enabled \
            --enforce-eager \
            --no-enable-prefix-caching \
            &>/tmp/bench_tail.log"

    log "Waiting for tail node on port ${TAIL_HOST_PORT}..."
    local elapsed=0
    while ! curl -sf --noproxy localhost "http://localhost:${TAIL_HOST_PORT}/health" >/dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ "$elapsed" -ge "$HEALTH_TIMEOUT" ]; then
            die "Tail node health check timed out"
        fi
    done
    log "Tail node ready!"
    sleep 5
}

# =========================== MOLINK v0.19 ===================================
# External code at /gxq/molink-measurement/MoLink, vllm conda env (vllm 0.19.0)

start_molink_v019() {
    log "=== MoLink v0.19 head node (layers 0-21) ==="
    docker exec -d -e PYTHONPATH="${V019_PYTHONPATH}" \
        -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 \
        "$C1_NAME" bash -c "\
        ${V019_PYTHON} -m molinkv1.entrypoints.api_server \
            --model ${MODEL_PATH} \
            --molink-grpc-port ${MOLINK_GRPC_HEAD} \
            --molink-start-layer 0 \
            --molink-end-layer 21 \
            --port ${HEAD_HTTP_PORT} \
            --max-model-len ${MAX_MODEL_LEN} \
            --molink-max-concurrent-batches 2 \
            --enforce-eager \
            --no-enable-prefix-caching \
            &>/tmp/bench_head.log"

    wait_for_health "http://localhost:${HEAD_HOST_PORT}/health" 300
    log "Head node ready. Waiting 20s for gRPC stabilization..."
    sleep 20

    log "=== MoLink v0.19 tail node (layers 21-end) ==="
    docker exec -d -e PYTHONPATH="${V019_PYTHONPATH}" \
        -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 \
        "$C2_NAME" bash -c "\
        ${V019_PYTHON} -m molinkv1.entrypoints.api_server \
            --model ${MODEL_PATH} \
            --molink-grpc-port ${MOLINK_GRPC_TAIL} \
            --molink-start-layer 21 \
            --molink-end-layer -1 \
            --port ${TAIL_HTTP_PORT} \
            --max-model-len ${MAX_MODEL_LEN} \
            --molink-max-concurrent-batches 2 \
            --molink-initial-peer ${C1_IP}:${MOLINK_GRPC_HEAD} \
            --enforce-eager \
            --no-enable-prefix-caching \
            &>/tmp/bench_tail.log"

    log "Waiting for tail node on port ${TAIL_HOST_PORT}..."
    local elapsed=0
    while ! curl -sf --noproxy localhost "http://localhost:${TAIL_HOST_PORT}/health" >/dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ "$elapsed" -ge "$HEALTH_TIMEOUT" ]; then
            die "Tail node health check timed out"
        fi
    done
    log "Tail node ready!"
    sleep 5
}

# =========================== BENCHMARK LOOP =================================

run_benchmarks_for() {
    local system="$1"
    local net_label="$2"

    local results_subdir="${RESULTS_DIR}/${system}/${net_label}"
    mkdir -p "$results_subdir"

    for rps in "${RPS_VALUES[@]}"; do
        local outdir="${results_subdir}/rps${rps}"
        mkdir -p "$outdir"

        log "--- [${system}] [${net_label}] rps=${rps} ---"

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

        log "Done: rps=${rps} -> ${outdir}/result.json"
        sleep "$COOLDOWN"
    done
}

# =========================== MAIN ===========================================

main() {
    local systems=()
    if [ $# -eq 0 ]; then
        systems=(molink011 molink019)
    else
        systems=("$@")
    fi

    log "==========================================="
    log " MoLink v0.11 vs v0.19 comparison"
    log " Systems      : ${systems[*]}"
    log " Network conds: ${NETWORK_CONDITIONS[*]}"
    log " RPS values   : ${RPS_VALUES[*]}"
    log " Duration     : ${DURATION}s per run"
    log " Results dir  : ${RESULTS_DIR}"
    log "==========================================="

    mkdir -p "$RESULTS_DIR"

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

    cleanup
    ensure_network
    start_containers

    for system in "${systems[@]}"; do
        for net_cond in "${NETWORK_CONDITIONS[@]}"; do
            bw="${net_cond%%,*}"
            delay="${net_cond##*,}"
            if [ "$net_cond" = "none" ]; then
                net_label="no_limit"
            else
                net_label="bw${bw}_delay${delay}"
            fi

            log "=========== ${system} | ${net_label} ==========="

            if [ "$net_cond" = "none" ]; then
                reset_network
            else
                setup_network "$bw" "$delay"
            fi
            stop_services

            case "$system" in
                molink011) start_molink_v011 ;;
                molink019) start_molink_v019 ;;
                *) die "Unknown system: $system" ;;
            esac

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
