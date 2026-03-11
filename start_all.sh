#!/bin/bash
# ═══════════════════════════════════════════════════════════
# METIS Full System Launcher — Phase 18: AGI Trinity
# ═══════════════════════════════════════════════════════════
#
# Services:
#   1. Training Dashboard   (port 8501) — GPU/training monitor
#   2. DPO Monitor          (port 5555) — legacy metrics API
#   3. Dreaming Daemon       (background) — autonomous gap training
#
# Usage:
#   bash start_all.sh          — start all services
#   bash start_all.sh stop     — stop all services
#   bash start_all.sh status   — check service status
#
set -euo pipefail

# ── Environment ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

METIS_VENV="${METIS_VENV:-$(cd "$SCRIPT_DIR" && python3 -c 'import sys; print(sys.prefix)' 2>/dev/null || echo "$SCRIPT_DIR/.venv")}"
if [ -f "$METIS_VENV/bin/activate" ]; then
    source "$METIS_VENV/bin/activate"
fi
NVIDIA_CUDA_RT="$(python3 -c 'import nvidia.cuda_runtime; import os; print(os.path.join(os.path.dirname(nvidia.cuda_runtime.__file__), "lib"))' 2>/dev/null || true)"
export LD_LIBRARY_PATH="${NVIDIA_CUDA_RT:+$NVIDIA_CUDA_RT:}${LD_LIBRARY_PATH:-}"
export TORCHDYNAMO_DISABLE=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-$(which ptxas 2>/dev/null || echo /usr/local/cuda/bin/ptxas)}"

PYTHON="${METIS_VENV}/bin/python3"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# ── Configuration ──
MODEL_PATH="experiment_output_dpo_balanced/metis_dpo_cognitive"
GAP_STORAGE="data/knowledge_gaps.json"
DREAM_OUTPUT="experiment_output_dreams"

# ── PID files ──
PID_DASHBOARD="$LOG_DIR/dashboard.pid"
PID_MONITOR="$LOG_DIR/monitor.pid"
PID_DAEMON="$LOG_DIR/daemon.pid"

# ── Helper functions ──
is_running() {
    local pidfile="$1"
    if [ -f "$pidfile" ]; then
        local pid
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        rm -f "$pidfile"
    fi
    return 1
}

stop_service() {
    local name="$1"
    local pidfile="$2"
    if is_running "$pidfile"; then
        local pid
        pid=$(cat "$pidfile")
        echo "  Stopping $name (PID $pid)..."
        kill "$pid" 2>/dev/null || true
        sleep 1
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
        rm -f "$pidfile"
        echo "  $name stopped."
    else
        echo "  $name not running."
    fi
}

# ── Commands ──
case "${1:-start}" in

stop)
    echo "═══ Stopping METIS Services ═══"
    stop_service "Training Dashboard" "$PID_DASHBOARD"
    stop_service "DPO Monitor" "$PID_MONITOR"
    stop_service "Dreaming Daemon" "$PID_DAEMON"
    rm -f /tmp/metis_daemon.lock
    echo "All services stopped."
    exit 0
    ;;

status)
    echo "═══ METIS Service Status ═══"
    for svc_pid in "Training Dashboard:$PID_DASHBOARD" "DPO Monitor:$PID_MONITOR" "Dreaming Daemon:$PID_DAEMON"; do
        name="${svc_pid%%:*}"
        pidfile="${svc_pid##*:}"
        if is_running "$pidfile"; then
            pid=$(cat "$pidfile")
            echo "  ✓ $name — running (PID $pid)"
        else
            echo "  ✗ $name — stopped"
        fi
    done
    echo ""
    echo "GPU:"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo "  nvidia-smi unavailable"
    exit 0
    ;;

start)
    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║   METIS Full System — Phase 18: AGI Trinity ║"
    echo "╚══════════════════════════════════════════════╝"
    echo ""

    # ── 1. Training Dashboard (port 8501) ──
    if is_running "$PID_DASHBOARD"; then
        echo "[1/3] Training Dashboard — already running (PID $(cat "$PID_DASHBOARD"))"
    else
        echo "[1/3] Starting Training Dashboard (port 8501)..."
        nohup $PYTHON dashboard.py --port 8501 > "$LOG_DIR/dashboard.log" 2>&1 &
        echo $! > "$PID_DASHBOARD"
        echo "       PID $(cat "$PID_DASHBOARD") → http://localhost:8501"
    fi

    # ── 2. DPO Monitor (port 5555) ──
    if is_running "$PID_MONITOR"; then
        echo "[2/3] DPO Monitor — already running (PID $(cat "$PID_MONITOR"))"
    else
        echo "[2/3] Starting DPO Monitor (port 5555)..."
        nohup $PYTHON monitor/app.py > "$LOG_DIR/monitor.log" 2>&1 &
        echo $! > "$PID_MONITOR"
        echo "       PID $(cat "$PID_MONITOR") → http://localhost:5555"
    fi

    # ── 3. Dreaming Daemon ──
    if is_running "$PID_DAEMON"; then
        echo "[3/3] Dreaming Daemon — already running (PID $(cat "$PID_DAEMON"))"
    else
        echo "[3/3] Starting Dreaming Daemon..."
        echo "       Gap storage: $GAP_STORAGE"
        echo "       Base model:  $MODEL_PATH"
        echo "       Output:      $DREAM_OUTPUT"
        nohup $PYTHON -m metis.daemon \
            --gap-path "$GAP_STORAGE" \
            --base-model "$MODEL_PATH" \
            --output-dir "$DREAM_OUTPUT" \
            --interval 30 \
            --gpu-threshold 10 \
            --min-gaps 5 \
            > "$LOG_DIR/daemon.log" 2>&1 &
        echo $! > "$PID_DAEMON"
        echo "       PID $(cat "$PID_DAEMON") → logs: logs/daemon.log"
    fi

    echo ""
    echo "═══ All services started ═══"
    echo ""
    echo "  Dashboard:  http://localhost:8501"
    echo "  Monitor:    http://localhost:5555"
    echo "  Daemon log: tail -f logs/daemon.log"
    echo ""
    echo "  Stop all:   bash start_all.sh stop"
    echo "  Status:     bash start_all.sh status"
    echo "  CLI REPL:   $PYTHON tools/run_metis_cli.py"
    echo ""
    ;;

*)
    echo "Usage: bash start_all.sh [start|stop|status]"
    exit 1
    ;;
esac
