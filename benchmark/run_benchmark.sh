#!/usr/bin/env bash
# ===========================================================================
# run_benchmark.sh — Automated benchmark runner for vLLM vs MoLink
#
# What it does:
#   1. Launch Docker containers on a custom network
#   2. Apply tc/netem network shaping (bandwidth + latency)
#   3. Start either MoLink or vLLM (with Ray)
#   4. Run the Python benchmark client for each RPS value
#   5. Collect JSON results and tear down
#
# Usage:
#   PP_SIZE=2 TP_SIZE=1 ./run_benchmark.sh              # PP=2 (default)
#   PP_SIZE=2 TP_SIZE=2 ./run_benchmark.sh              # PP=2 + TP=2
#   PP_SIZE=3 TP_SIZE=1 ./run_benchmark.sh              # PP=3
#   PP_SIZE=2 TP_SIZE=2 ./run_benchmark.sh molink       # only MoLink
#   PP_SIZE=2 TP_SIZE=2 ./run_benchmark.sh molink vllm  # both
# ===========================================================================
set -euo pipefail

# =========================== CONFIGURATION ==================================

# Pipeline / tensor parallelism (override via env vars)
PP_SIZE=${PP_SIZE:-2}
TP_SIZE=${TP_SIZE:-1}

# Validate GPU requirements (GPUs 1-4 available, GPU 0 reserved for host)
AVAILABLE_GPUS=4
GPUS_NEEDED=$((PP_SIZE * TP_SIZE))
if [ "$GPUS_NEEDED" -gt "$AVAILABLE_GPUS" ]; then
    echo "ERROR: PP=${PP_SIZE} × TP=${TP_SIZE} requires ${GPUS_NEEDED} GPUs, only ${AVAILABLE_GPUS} available" >&2
    exit 1
fi

# Docker
DOCKER_IMAGE="molink:0.2"
DOCKER_NETWORK="molink"

# Container names / IPs
C1_NAME="bench-head"
C1_IP="172.26.0.10"
C2_NAME="bench-middle"       # only used when PP_SIZE=3
C2_IP="172.26.0.12"
C3_NAME="bench-tail"
C3_IP="172.26.0.11"

# Compute GPU device lists (Docker --gpus "device=X,Y,...")
_gpu_list() {
    local base=$1 count=$2
    local result=$base i=1
    while [ "$i" -lt "$count" ]; do
        result="${result},$((base + i))"
        i=$((i + 1))
    done
    echo "$result"
}
# GPU allocation: contiguous groups starting from GPU 0
# PP=2: head=[0..TP-1], tail=[TP .. 2*TP-1]
# PP=3: head=[0..TP-1], middle=[TP .. 2*TP-1], tail=[2*TP .. 3*TP-1]
GPU_HEAD=$(_gpu_list 0 "$TP_SIZE")
GPU_MIDDLE=$(_gpu_list "$TP_SIZE" "$TP_SIZE")
if [ "$PP_SIZE" -eq 3 ]; then
    GPU_TAIL=$(_gpu_list $((2 * TP_SIZE)) "$TP_SIZE")
else
    GPU_TAIL=$(_gpu_list "$TP_SIZE" "$TP_SIZE")
fi

# CUDA_VISIBLE_DEVICES string inside container (Docker remaps device IDs to 0..N)
CUDA_DEVS=$(seq -s, 0 $((TP_SIZE - 1)))

# Layer split for MoLink (Qwen3-14B: ~40 layers)
if [ "$PP_SIZE" -eq 3 ]; then
    HEAD_END_LAYER=14
    MIDDLE_START=14
    MIDDLE_END=28
    TAIL_START=28
else
    HEAD_END_LAYER=21
    TAIL_START=21
fi

# Paths inside containers
MODEL_PATH="/gxq/Qwen3-14B"
HOST_TOKENIZER_PATH="/home/emnets-2/gxq/Qwen3-14B"
MOLINK_CODE="/gxq/molink-measurement/MoLink"
MOLINK_PYTHON="/opt/conda/envs/vllm19/bin/python"
VLLM_BIN="/opt/conda/envs/vllm19/bin/vllm"
RAY_BIN="/opt/conda/envs/vllm19/bin/ray"
VLLM_SITE_PACKAGES="/opt/conda/envs/vllm19/lib/python3.12/site-packages/vllm"
VLLM_SOURCE="/gxq/molink-measurement/vllm/vllm"

# Ports
HEAD_HTTP_PORT=8080
HEAD_HOST_PORT=8080
MIDDLE_HTTP_PORT=9096
MIDDLE_HOST_PORT=9096
TAIL_HTTP_PORT=9095
TAIL_HOST_PORT=9095
MOLINK_GRPC_HEAD=50061
MOLINK_GRPC_MIDDLE=50062
MOLINK_GRPC_TAIL=50063
RAY_PORT=6379

# Network conditions to test: "bandwidth,latency"
NETWORK_CONDITIONS=(
    # "none"
    "1gbit,10ms"
    # "5gbit,10ms"
    # "1gbit,20ms"
    # "1gbit,30ms"
    # "500mbit,10ms"
    # 100mbit,10ms
)

# Trace parameters
INPUT_TOKENS=1024
OUTPUT_TOKENS=512
RPS_VALUES=(3)
DURATION=30
COOLDOWN=10
MAX_MODEL_LEN=4096

# MoLink pipeline config
MAX_CONCURRENT_BATCHES=${MOLINK_MAX_CONCURRENT_BATCHES:-2}

# Results
RESULTS_ROOT="/home/emnets-2/gxq/molink-measurement/MoLink/benchmark/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${RESULTS_ROOT}/${TIMESTAMP}"
BENCHMARK_CLIENT="/home/emnets-2/gxq/molink-measurement/MoLink/benchmark/benchmark_client.py"

HEALTH_TIMEOUT=300

# =========================== LOGGING ========================================

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
warn() { log "WARN: $*" >&2; }
die()  { log "ERROR: $*" >&2; exit 1; }

# =========================== CLEANUP ========================================

cleanup() {
    log "Cleaning up containers and processes..."
    docker rm -f "$C1_NAME" "$C2_NAME" "$C3_NAME" 2>/dev/null || true
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
    log "Starting containers (PP=${PP_SIZE} TP=${TP_SIZE})..."

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

    if [ "$PP_SIZE" -eq 3 ]; then
        docker run -d --name "$C2_NAME" \
            -v /mnt/disk1-16/gxq:/data \
            -v /home/emnets-2/gxq:/gxq \
            --gpus "\"device=${GPU_MIDDLE}\"" \
            --shm-size=64g \
            --cap-add=NET_ADMIN \
            --network "$DOCKER_NETWORK" \
            --ip "$C2_IP" \
            -p "${MIDDLE_HOST_PORT}:${MIDDLE_HTTP_PORT}" \
            "$DOCKER_IMAGE" \
            sleep infinity
    fi

    docker run -d --name "$C3_NAME" \
        -v /mnt/disk1-16/gxq:/data \
        -v /home/emnets-2/gxq:/gxq \
        --gpus "\"device=${GPU_TAIL}\"" \
        --shm-size=64g \
        --cap-add=NET_ADMIN \
        --network "$DOCKER_NETWORK" \
        --ip "$C3_IP" \
        -p "${TAIL_HOST_PORT}:${TAIL_HTTP_PORT}" \
        "$DOCKER_IMAGE" \
        sleep infinity

    if [ "$PP_SIZE" -eq 3 ]; then
        log "Containers started ($C1_NAME=$C1_IP [$GPU_HEAD]  $C2_NAME=$C2_IP [$GPU_MIDDLE]  $C3_NAME=$C3_IP [$GPU_TAIL])"
    else
        log "Containers started ($C1_NAME=$C1_IP [$GPU_HEAD]  $C3_NAME=$C3_IP [$GPU_TAIL])"
    fi
    sleep 5
}

# =========================== NETWORK SHAPING ================================

_setup_tc() {
    local container="$1" bw="$2" delay="$3"
    shift 3
    docker exec "$container" bash -c "tc qdisc del dev eth0 root 2>/dev/null || true"
    docker exec "$container" bash -c "tc qdisc add dev eth0 root handle 1: htb default 30"
    docker exec "$container" bash -c "tc class add dev eth0 parent 1: classid 1:1 htb rate ${bw}"
    docker exec "$container" bash -c "tc qdisc add dev eth0 parent 1:1 netem delay ${delay}"
    local prio=1
    for ip in "$@"; do
        docker exec "$container" bash -c \
            "tc filter add dev eth0 parent 1: protocol ip prio ${prio} u32 match ip dst ${ip} flowid 1:1"
        prio=$((prio + 1))
    done
}

setup_network() {
    local bw="$1" delay="$2"
    log "Applying network shaping: ${bw} / ${delay} (PP=${PP_SIZE} TP=${TP_SIZE})"

    if [ "$PP_SIZE" -eq 3 ]; then
        _setup_tc "$C1_NAME" "$bw" "$delay" "$C2_IP" "$C3_IP"
        _setup_tc "$C2_NAME" "$bw" "$delay" "$C1_IP" "$C3_IP"
        _setup_tc "$C3_NAME" "$bw" "$delay" "$C1_IP" "$C2_IP"
    else
        _setup_tc "$C1_NAME" "$bw" "$delay" "$C3_IP"
        _setup_tc "$C3_NAME" "$bw" "$delay" "$C1_IP"
    fi

    log "Network shaping applied."
}

reset_network() {
    log "Removing network shaping (no limit)..."
    docker exec "$C1_NAME" bash -c "tc qdisc del dev eth0 root 2>/dev/null || true"
    docker exec "$C3_NAME" bash -c "tc qdisc del dev eth0 root 2>/dev/null || true"
    if [ "$PP_SIZE" -eq 3 ]; then
        docker exec "$C2_NAME" bash -c "tc qdisc del dev eth0 root 2>/dev/null || true"
    fi
    log "Network shaping removed."
}

# =========================== SERVICE MANAGEMENT =============================

_pkill_services() {
    local container="$1"
    docker exec "$container" bash -c "\
        pkill -f 'python.*molinkv1' 2>/dev/null || true; \
        pkill -f 'python.*vllm' 2>/dev/null || true; \
        pkill -f 'ray::' 2>/dev/null || true; \
        pkill -f 'raylet' 2>/dev/null || true; \
        pkill -f 'VLLM' 2>/dev/null || true" || true
}

stop_services() {
    log "Stopping services inside containers..."
    _pkill_services "$C1_NAME"
    _pkill_services "$C3_NAME"
    if [ "$PP_SIZE" -eq 3 ]; then
        _pkill_services "$C2_NAME"
    fi
    sleep 5
}

deploy_instrumented_vllm() {
    log "Deploying instrumented vLLM code to containers..."
    for container in "$C1_NAME" "$C3_NAME"; do
        docker exec "$container" bash -c "\
            cp ${VLLM_SOURCE}/v1/executor/ray_executor.py ${VLLM_SITE_PACKAGES}/v1/executor/ray_executor.py && \
            cp ${VLLM_SOURCE}/v1/executor/ray_utils.py ${VLLM_SITE_PACKAGES}/v1/executor/ray_utils.py && \
            find ${VLLM_SITE_PACKAGES} -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null; \
            rm -f /tmp/vllm_metrics.json /tmp/vllm_worker_metrics.json 2>/dev/null || true; \
            echo done"
    done
    if [ "$PP_SIZE" -eq 3 ]; then
        docker exec "$C2_NAME" bash -c "\
            cp ${VLLM_SOURCE}/v1/executor/ray_executor.py ${VLLM_SITE_PACKAGES}/v1/executor/ray_executor.py && \
            cp ${VLLM_SOURCE}/v1/executor/ray_utils.py ${VLLM_SITE_PACKAGES}/v1/executor/ray_utils.py && \
            find ${VLLM_SITE_PACKAGES} -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null; \
            rm -f /tmp/vllm_metrics.json /tmp/vllm_worker_metrics.json 2>/dev/null || true; \
            echo done"
    fi
    log "Instrumented vLLM code deployed."
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
            log "=== $C3_NAME logs ==="
            docker exec "$C3_NAME" bash -c "tail -20 /tmp/bench_tail.log" 2>/dev/null || true
            if [ "$PP_SIZE" -eq 3 ]; then
                log "=== $C2_NAME logs ==="
                docker exec "$C2_NAME" bash -c "tail -20 /tmp/bench_middle.log" 2>/dev/null || true
            fi
            die "Health check timed out after ${timeout}s"
        fi
        if [ $((elapsed % 30)) -eq 0 ]; then
            log "  ... still waiting (${elapsed}s/${timeout}s)"
        fi
    done
    log "Service is healthy!"
}

# ---------- MoLink PP=2 ----------

start_molink_pp2() {
    # Clear __pycache__ and stale metrics files
    for container in "$C1_NAME" "$C3_NAME"; do
        docker exec "$container" bash -c \
            "find ${MOLINK_CODE} -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true; \
             rm -f /tmp/molink_metrics_*.json 2>/dev/null || true" || true
    done

    log "Starting MoLink head node (layers 0-${HEAD_END_LAYER}, PP=2 TP=${TP_SIZE})..."
    docker exec -d -e PYTHONPATH="${MOLINK_CODE}" \
        -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 \
        "$C1_NAME" bash -c "\
        ${MOLINK_PYTHON} -m molinkv1.entrypoints.api_server \
            --model ${MODEL_PATH} \
            --max-model-len ${MAX_MODEL_LEN} \
            --molink-max-concurrent-batches ${MAX_CONCURRENT_BATCHES} \
            --tensor-parallel-size ${TP_SIZE} \
            --enforce-eager \
            --no-enable-prefix-caching \
            --molink-grpc-port ${MOLINK_GRPC_HEAD} \
            --molink-start-layer 0 \
            --molink-end-layer ${HEAD_END_LAYER} \
            --molink-enable-metrics \
            --port ${HEAD_HTTP_PORT} \
            &>/tmp/bench_head.log"

    wait_for_health "http://localhost:${HEAD_HOST_PORT}/health" 300
    log "Head node ready. Waiting 20s for gRPC stabilization..."
    sleep 20

    log "Starting MoLink tail node (layers ${TAIL_START}-end, PP=2 TP=${TP_SIZE})..."
    docker exec -d -e PYTHONPATH="${MOLINK_CODE}" \
        -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 \
        "$C3_NAME" bash -c "\
        ${MOLINK_PYTHON} -m molinkv1.entrypoints.api_server \
            --model ${MODEL_PATH} \
            --max-model-len ${MAX_MODEL_LEN} \
            --molink-max-concurrent-batches ${MAX_CONCURRENT_BATCHES} \
            --tensor-parallel-size ${TP_SIZE} \
            --enforce-eager \
            --no-enable-prefix-caching \
            --molink-grpc-port ${MOLINK_GRPC_TAIL} \
            --molink-start-layer ${TAIL_START} \
            --molink-end-layer -1 \
            --molink-enable-metrics \
            --port ${TAIL_HTTP_PORT} \
            --molink-initial-peer ${C1_IP}:${MOLINK_GRPC_HEAD} \
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

# ---------- MoLink PP=3 ----------

start_molink_pp3() {
    log "Starting MoLink head node (layers 0-${HEAD_END_LAYER}, PP=3 TP=${TP_SIZE})..."
    docker exec -d -e PYTHONPATH="${MOLINK_CODE}" \
        -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 \
        "$C1_NAME" bash -c "\
        ${MOLINK_PYTHON} -m molinkv1.entrypoints.api_server \
            --model ${MODEL_PATH} \
            --max-model-len ${MAX_MODEL_LEN} \
            --molink-max-concurrent-batches ${MAX_CONCURRENT_BATCHES} \
            --tensor-parallel-size ${TP_SIZE} \
            --enforce-eager \
            --no-enable-prefix-caching \
            --molink-grpc-port ${MOLINK_GRPC_HEAD} \
            --molink-start-layer 0 \
            --molink-end-layer ${HEAD_END_LAYER} \
            --port ${HEAD_HTTP_PORT} \
            &>/tmp/bench_head.log"

    wait_for_health "http://localhost:${HEAD_HOST_PORT}/health" 300
    log "Head node ready. Waiting 20s for gRPC stabilization..."
    sleep 20

    log "Starting MoLink middle node (layers ${MIDDLE_START}-${MIDDLE_END}, PP=3 TP=${TP_SIZE})..."
    docker exec -d -e PYTHONPATH="${MOLINK_CODE}" \
        -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 \
        "$C2_NAME" bash -c "\
        ${MOLINK_PYTHON} -m molinkv1.entrypoints.api_server \
            --model ${MODEL_PATH} \
            --max-model-len ${MAX_MODEL_LEN} \
            --molink-max-concurrent-batches ${MAX_CONCURRENT_BATCHES} \
            --tensor-parallel-size ${TP_SIZE} \
            --enforce-eager \
            --no-enable-prefix-caching \
            --molink-grpc-port ${MOLINK_GRPC_MIDDLE} \
            --molink-start-layer ${MIDDLE_START} \
            --molink-end-layer ${MIDDLE_END} \
            --port ${MIDDLE_HTTP_PORT} \
            --molink-initial-peer ${C1_IP}:${MOLINK_GRPC_HEAD} \
            &>/tmp/bench_middle.log"

    log "Waiting for middle node on port ${MIDDLE_HOST_PORT}..."
    local elapsed=0
    while ! curl -sf --noproxy localhost "http://localhost:${MIDDLE_HOST_PORT}/health" >/dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ "$elapsed" -ge "$HEALTH_TIMEOUT" ]; then
            die "Middle node health check timed out"
        fi
    done
    log "Middle node ready!"
    sleep 5

    log "Starting MoLink tail node (layers ${TAIL_START}-end, PP=3 TP=${TP_SIZE})..."
    docker exec -d -e PYTHONPATH="${MOLINK_CODE}" \
        -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 \
        "$C3_NAME" bash -c "\
        ${MOLINK_PYTHON} -m molinkv1.entrypoints.api_server \
            --model ${MODEL_PATH} \
            --max-model-len ${MAX_MODEL_LEN} \
            --molink-max-concurrent-batches ${MAX_CONCURRENT_BATCHES} \
            --tensor-parallel-size ${TP_SIZE} \
            --enforce-eager \
            --no-enable-prefix-caching \
            --molink-grpc-port ${MOLINK_GRPC_TAIL} \
            --molink-start-layer ${TAIL_START} \
            --molink-end-layer -1 \
            --port ${TAIL_HTTP_PORT} \
            --molink-initial-peer ${C2_IP}:${MOLINK_GRPC_MIDDLE} \
            &>/tmp/bench_tail.log"

    log "Waiting for tail node on port ${TAIL_HOST_PORT}..."
    elapsed=0
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

start_molink() {
    if [ "$PP_SIZE" -eq 3 ]; then
        start_molink_pp3
    else
        start_molink_pp2
    fi
}

# ---------- vLLM + Ray ----------

start_vllm() {
    deploy_instrumented_vllm

    log "Starting Ray head in $C1_NAME (${TP_SIZE} GPU(s))..."
    docker exec -d "$C1_NAME" bash -c "\
        CUDA_VISIBLE_DEVICES=${CUDA_DEVS} VLLM_WORKER_METRICS_FILE=/tmp/vllm_worker_metrics.json \
        ${RAY_BIN} start --head \
            --node-ip-address=${C1_IP} \
            --port=${RAY_PORT} --num-gpus=${TP_SIZE}"
    sleep 8

    if [ "$PP_SIZE" -eq 3 ]; then
        log "Starting Ray worker in $C2_NAME (${TP_SIZE} GPU(s))..."
        docker exec -d "$C2_NAME" bash -c "\
            CUDA_VISIBLE_DEVICES=${CUDA_DEVS} VLLM_WORKER_METRICS_FILE=/tmp/vllm_worker_metrics.json \
            ${RAY_BIN} start \
                --address=${C1_IP}:${RAY_PORT} --num-gpus=${TP_SIZE}"
        sleep 8
    fi

    log "Starting Ray worker in $C3_NAME (${TP_SIZE} GPU(s))..."
    docker exec -d "$C3_NAME" bash -c "\
        CUDA_VISIBLE_DEVICES=${CUDA_DEVS} VLLM_WORKER_METRICS_FILE=/tmp/vllm_worker_metrics.json \
        ${RAY_BIN} start \
            --address=${C1_IP}:${RAY_PORT} --num-gpus=${TP_SIZE}"
    sleep 8

    log "Starting vLLM serve (PP=${PP_SIZE} TP=${TP_SIZE})..."
    docker exec -d "$C1_NAME" bash -c "\
        CUDA_VISIBLE_DEVICES=${CUDA_DEVS} VLLM_METRICS_FILE=/tmp/vllm_metrics.json \
        ${VLLM_BIN} serve \
            --model ${MODEL_PATH} \
            --port ${HEAD_HTTP_PORT} \
            --max-model-len ${MAX_MODEL_LEN} \
            --pipeline-parallel-size ${PP_SIZE} \
            --tensor-parallel-size ${TP_SIZE} \
            --distributed-executor-backend ray \
            --no-enable-prefix-caching \
            --enforce-eager"

    wait_for_health
}

# =========================== BENCHMARK LOOP =================================

collect_metrics() {
    local system="$1"
    local outdir="$2"

    log "Collecting ${system} metrics to ${outdir}/..."

    if [ "$system" = "molink" ]; then
        # Collect MoLink metrics from head and tail containers
        docker exec "$C1_NAME" bash -c "cat /tmp/molink_metrics_*.json 2>/dev/null || echo '{}'" \
            > "${outdir}/head_metrics.json" 2>/dev/null || true
        docker exec "$C3_NAME" bash -c "cat /tmp/molink_metrics_*.json 2>/dev/null || echo '{}'" \
            > "${outdir}/tail_metrics.json" 2>/dev/null || true
        if [ "$PP_SIZE" -eq 3 ]; then
            docker exec "$C2_NAME" bash -c "cat /tmp/molink_metrics_*.json 2>/dev/null || echo '{}'" \
                > "${outdir}/middle_metrics.json" 2>/dev/null || true
        fi
        # Collect HTTP metrics endpoint
        curl -sf --noproxy localhost "http://localhost:${HEAD_HOST_PORT}/molink_metrics" \
            > "${outdir}/molink_http_metrics.json" 2>/dev/null || true
    else
        # Collect vLLM/Ray metrics from head container
        docker exec "$C1_NAME" bash -c "cat /tmp/vllm_metrics.json 2>/dev/null || echo '{}'" \
            > "${outdir}/vllm_metrics.json" 2>/dev/null || true
        # Collect worker-side metrics from each container
        docker exec "$C1_NAME" bash -c "cat /tmp/vllm_worker_metrics.json 2>/dev/null || echo '{}'" \
            > "${outdir}/vllm_worker_head.json" 2>/dev/null || true
        docker exec "$C3_NAME" bash -c "cat /tmp/vllm_worker_metrics.json 2>/dev/null || echo '{}'" \
            > "${outdir}/vllm_worker_tail.json" 2>/dev/null || true
        if [ "$PP_SIZE" -eq 3 ]; then
            docker exec "$C2_NAME" bash -c "cat /tmp/vllm_worker_metrics.json 2>/dev/null || echo '{}'" \
                > "${outdir}/vllm_worker_middle.json" 2>/dev/null || true
        fi
    fi
}

run_benchmarks_for() {
    local system="$1"
    local net_label="$2"

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

        collect_metrics "$system" "$outdir"
        log "Done: rps=${rps} -> ${outdir}/result.json"
        sleep "$COOLDOWN"
    done
}

# =========================== MAIN ===========================================

main() {
    local systems=()
    if [ $# -eq 0 ]; then
        systems=(molink vllm)
    else
        systems=("$@")
    fi

    log "==========================================="
    log " Benchmark suite  PP=${PP_SIZE}  TP=${TP_SIZE}"
    log " GPUs needed  : ${GPUS_NEEDED} (${AVAILABLE_GPUS} available)"
    log " GPU devices  : head=[${GPU_HEAD}]  tail=[${GPU_TAIL}]"
    if [ "$PP_SIZE" -eq 3 ]; then
        log "                middle=[${GPU_MIDDLE}]"
    fi
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
  "pp_size": ${PP_SIZE},
  "tp_size": ${TP_SIZE},
  "gpus_needed": ${GPUS_NEEDED},
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

            log "=========== ${system} (PP=${PP_SIZE} TP=${TP_SIZE}) | ${net_label} ==========="

            if [ "$net_cond" = "none" ]; then
                reset_network
            else
                setup_network "$bw" "$delay"
            fi
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
    log " All benchmarks complete! (PP=${PP_SIZE} TP=${TP_SIZE})"
    log " Results: ${RESULTS_DIR}"
    log "==========================================="
}

main "$@"
