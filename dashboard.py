#!/usr/bin/env python3
"""
METIS Training Dashboard — Real-time monitoring for SFT/DPO pipeline.

Usage:
    python3 dashboard.py [--port 8501] [--log pipeline_v2_sft_dpo.log]

Features:
    - GPU utilization, temperature, memory, power
    - Training phase detection (SFT Warmup / DPO METIS / DPO Random / Eval)
    - Step progress with ETA
    - Live loss curve
    - Log tail (last 50 lines)
    - 3-second auto-refresh
"""

import argparse
import json
import os
import re
import subprocess
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# ─── Configuration ───────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
DEFAULT_LOG = BASE_DIR / "pipeline_v2_sft_dpo.log"
DEFAULT_OUTPUT_DIR = BASE_DIR / "experiment_output_7B_orca_v2_sft_dpo"


def get_gpu_stats() -> dict[str, Any]:
    """Query nvidia-smi for GPU stats."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu,power.draw,power.limit,memory.used,memory.total,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return {"error": "nvidia-smi failed"}
        
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) >= 7:
            return {
                "gpu_util": int(parts[0]) if parts[0] not in ("[N/A]", "N/A", "") else -1,
                "temp_c": int(parts[1]) if parts[1] not in ("[N/A]", "N/A", "") else -1,
                "power_w": float(parts[2]) if parts[2] not in ("[N/A]", "N/A", "") else -1,
                "power_cap_w": float(parts[3]) if parts[3] not in ("[N/A]", "N/A", "") else -1,
                "mem_used_mb": int(float(parts[4])) if parts[4] not in ("[N/A]", "N/A", "") else -1,
                "mem_total_mb": int(float(parts[5])) if parts[5] not in ("[N/A]", "N/A", "") else -1,
                "name": parts[6],
            }
        return {"error": f"unexpected format: {parts}"}
    except Exception as e:
        return {"error": str(e)}


def get_process_stats() -> dict[str, Any]:
    """Get training process CPU/memory info."""
    try:
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if "run_experiment" in line and "grep" not in line:
                parts = line.split()
                return {
                    "pid": int(parts[1]),
                    "cpu_pct": float(parts[2]),
                    "mem_pct": float(parts[3]),
                    "start_time": parts[8],
                    "elapsed": parts[9],
                    "running": True,
                }
        return {"running": False}
    except Exception as e:
        return {"running": False, "error": str(e)}


def parse_training_log(log_path: str) -> dict[str, Any]:
    """Parse training log for phase, progress, loss values."""
    result: dict[str, Any] = {
        "phase": "UNKNOWN",
        "step": 0,
        "total_steps": 0,
        "progress_pct": 0.0,
        "eta": "N/A",
        "speed": "N/A",
        "losses": [],
        "log_lines": [],
        "sft_done": False,
        "dpo_metis_done": False,
        "dpo_random_done": False,
        "eval_done": False,
    }

    if not os.path.exists(log_path):
        result["phase"] = "NO LOG FILE"
        return result

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except Exception:
        return result

    # Split on \r to get latest progress bar overwrites
    lines = raw.replace("\r", "\n").split("\n")
    # Keep meaningful lines for display
    display_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and len(stripped) > 3:
            display_lines.append(stripped)
    result["log_lines"] = display_lines[-60:]

    full_text = "\n".join(display_lines)

    # Detect phases completed
    if "SFT Warmup complete" in full_text or "SFT warmup" in full_text.lower() and "Training Group A" in full_text:
        result["sft_done"] = True
    if "Training Group A (METIS DPO)" in full_text and "Training Group B" in full_text:
        result["dpo_metis_done"] = True
    if "Training Group B (Random DPO)" in full_text and "PHASE 3" in full_text:
        result["dpo_random_done"] = True

    # Detect current phase
    if "PHASE 3" in full_text or "Evaluation" in full_text.split("PHASE 2")[-1] if "PHASE 2" in full_text else "":
        result["phase"] = "EVALUATION"
    elif "Training Group B" in full_text and not result.get("dpo_random_done"):
        result["phase"] = "DPO_RANDOM"
    elif "Training Group A" in full_text and not result.get("dpo_metis_done"):
        result["phase"] = "DPO_METIS"
    elif "[SFT Warmup]" in full_text:
        result["phase"] = "SFT_WARMUP"
    elif "PHASE 2" in full_text:
        result["phase"] = "INITIALIZING"

    # Parse progress bars: "  X%|...|  step/total [time<eta, speed]"
    progress_pattern = re.compile(
        r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([^\]]*)\]"
    )
    progress_matches = progress_pattern.findall("\n".join(display_lines[-30:]))
    if progress_matches:
        last = progress_matches[-1]
        result["progress_pct"] = int(last[0])
        result["step"] = int(last[1])
        result["total_steps"] = int(last[2])
        timing = last[3]
        # Parse "05:21<5:32:22, 321.66s/it"
        eta_match = re.search(r"<([^,]+)", timing)
        speed_match = re.search(r",\s*(.+)", timing)
        if eta_match:
            result["eta"] = eta_match.group(1).strip()
        if speed_match:
            result["speed"] = speed_match.group(1).strip()

    # Parse loss values: "{'loss': 1.234, 'grad_norm': ..., 'learning_rate': ..., 'epoch': ...}"
    loss_pattern = re.compile(r"\{'loss':\s*([\d.]+).*?'epoch':\s*([\d.]+)")
    for m in loss_pattern.finditer(full_text):
        result["losses"].append({
            "loss": float(m.group(1)),
            "epoch": float(m.group(2)),
        })

    # Also parse "{'train_loss': X, ...}" summary
    train_loss_pattern = re.compile(r"\{'train_loss':\s*([\d.]+)")
    for m in train_loss_pattern.finditer(full_text):
        result["losses"].append({
            "loss": float(m.group(1)),
            "epoch": -1,  # summary
            "is_summary": True,
        })

    return result


def get_checkpoint_info() -> list[dict[str, Any]]:
    """Check for saved checkpoints."""
    checkpoints = []
    for subdir in ["sft_warmup", "metis_dpo", "random_dpo"]:
        path = DEFAULT_OUTPUT_DIR / subdir
        if path.exists():
            items = sorted(path.iterdir())
            size_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)
            checkpoints.append({
                "name": subdir,
                "path": str(path),
                "files": len(list(path.rglob("*"))),
                "size_mb": round(size_mb, 1),
                "items": [i.name for i in items[:10]],
            })
    return checkpoints


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>METIS Training Dashboard</title>
<style>
  :root {
    --bg: #0a0e17; --card: #111827; --border: #1e293b;
    --green: #22c55e; --cyan: #06b6d4; --yellow: #eab308;
    --red: #ef4444; --purple: #a855f7; --blue: #3b82f6;
    --text: #e2e8f0; --dim: #64748b; --bright: #f8fafc;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: var(--bg); color: var(--text);
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
    font-size: 13px; line-height: 1.5;
    padding: 16px; min-height: 100vh;
  }
  .header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 20px; background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; margin-bottom: 16px;
  }
  .header h1 {
    font-size: 18px; color: var(--cyan);
    text-shadow: 0 0 20px rgba(6,182,212,0.3);
  }
  .header .status {
    display: flex; gap: 12px; align-items: center;
  }
  .header .dot {
    width: 8px; height: 8px; border-radius: 50%;
    display: inline-block; margin-right: 4px;
  }
  .dot.green { background: var(--green); box-shadow: 0 0 8px var(--green); }
  .dot.red { background: var(--red); box-shadow: 0 0 8px var(--red); }
  .dot.yellow { background: var(--yellow); box-shadow: 0 0 8px var(--yellow); }

  .grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 16px; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px; }
  .grid-full { margin-bottom: 16px; }

  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px;
  }
  .card h2 {
    font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px;
    color: var(--dim); margin-bottom: 12px;
    padding-bottom: 8px; border-bottom: 1px solid var(--border);
  }
  .metric { margin-bottom: 8px; }
  .metric .label { color: var(--dim); font-size: 11px; }
  .metric .value { font-size: 22px; font-weight: 700; color: var(--bright); }
  .metric .value.green { color: var(--green); }
  .metric .value.cyan { color: var(--cyan); }
  .metric .value.yellow { color: var(--yellow); }
  .metric .value.red { color: var(--red); }
  .metric .value.purple { color: var(--purple); }

  .progress-container {
    background: #1a1a2e; border-radius: 6px; height: 28px;
    overflow: hidden; position: relative; margin: 8px 0;
  }
  .progress-bar {
    height: 100%; border-radius: 6px;
    background: linear-gradient(90deg, var(--cyan), var(--blue));
    transition: width 0.5s ease; position: relative;
    box-shadow: 0 0 15px rgba(6,182,212,0.4);
  }
  .progress-text {
    position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    font-size: 12px; font-weight: 600; color: var(--bright);
    text-shadow: 0 1px 3px rgba(0,0,0,0.8);
  }

  .pipeline-stages {
    display: flex; gap: 4px; margin: 12px 0;
  }
  .stage {
    flex: 1; padding: 8px; text-align: center; border-radius: 6px;
    font-size: 11px; font-weight: 600; position: relative;
    border: 1px solid var(--border);
  }
  .stage.done { background: rgba(34,197,94,0.15); border-color: var(--green); color: var(--green); }
  .stage.active { background: rgba(6,182,212,0.15); border-color: var(--cyan); color: var(--cyan);
    animation: pulse 2s infinite; }
  .stage.pending { background: rgba(100,116,139,0.1); color: var(--dim); }
  @keyframes pulse {
    0%, 100% { box-shadow: 0 0 5px rgba(6,182,212,0.2); }
    50% { box-shadow: 0 0 20px rgba(6,182,212,0.5); }
  }

  .gauge-row { display: flex; gap: 12px; margin-bottom: 8px; }
  .gauge {
    flex: 1; background: #1a1a2e; border-radius: 4px; height: 8px;
    overflow: hidden;
  }
  .gauge-fill { height: 100%; border-radius: 4px; transition: width 0.3s ease; }

  canvas { width: 100% !important; height: 200px !important; }

  .log-container {
    background: #050810; border: 1px solid var(--border); border-radius: 8px;
    padding: 12px; max-height: 300px; overflow-y: auto;
    font-size: 11px; line-height: 1.6;
  }
  .log-line { white-space: pre-wrap; word-break: break-all; }
  .log-line.info { color: var(--cyan); }
  .log-line.warn { color: var(--yellow); }
  .log-line.error { color: var(--red); }
  .log-line.progress { color: var(--green); }
  .log-line.dim { color: var(--dim); }

  .timestamp { color: var(--dim); font-size: 11px; }
  .chip {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 10px; font-weight: 700; letter-spacing: 0.5px;
  }
  .chip.sft { background: rgba(168,85,247,0.2); color: var(--purple); }
  .chip.dpo { background: rgba(6,182,212,0.2); color: var(--cyan); }
  .chip.eval { background: rgba(234,179,8,0.2); color: var(--yellow); }
  .chip.done { background: rgba(34,197,94,0.2); color: var(--green); }

  .checkpoint-list { display: flex; gap: 8px; flex-wrap: wrap; }
  .ckpt {
    padding: 6px 12px; background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.3); border-radius: 6px;
    font-size: 11px;
  }
  .ckpt .size { color: var(--dim); }

  @media (max-width: 900px) {
    .grid { grid-template-columns: 1fr; }
    .grid-2 { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
  <div class="header">
    <h1>METIS Training Dashboard</h1>
    <div class="status">
      <span id="proc-status"><span class="dot yellow"></span>Loading...</span>
      <span class="timestamp" id="last-update"></span>
    </div>
  </div>

  <!-- Pipeline Stages -->
  <div class="grid-full">
    <div class="card">
      <h2>Pipeline Stages</h2>
      <div class="pipeline-stages">
        <div class="stage pending" id="stage-sft">SFT Warmup</div>
        <div class="stage pending" id="stage-arrow1">&rarr;</div>
        <div class="stage pending" id="stage-dpo-m">DPO METIS</div>
        <div class="stage pending" id="stage-arrow2">&rarr;</div>
        <div class="stage pending" id="stage-dpo-r">DPO Random</div>
        <div class="stage pending" id="stage-arrow3">&rarr;</div>
        <div class="stage pending" id="stage-eval">Evaluation</div>
      </div>
      <div class="progress-container">
        <div class="progress-bar" id="main-progress" style="width:0%"></div>
        <div class="progress-text" id="main-progress-text">0%</div>
      </div>
      <div style="display:flex; justify-content:space-between; color:var(--dim); font-size:11px;">
        <span id="step-info">Step: 0/0</span>
        <span id="speed-info">Speed: N/A</span>
        <span id="eta-info">ETA: N/A</span>
      </div>
    </div>
  </div>

  <!-- GPU + Process + Phase Info -->
  <div class="grid">
    <div class="card">
      <h2>GPU Status</h2>
      <div class="metric">
        <div class="label">GPU Name</div>
        <div class="value cyan" id="gpu-name" style="font-size:14px;">-</div>
      </div>
      <div class="metric">
        <div class="label">Utilization</div>
        <div class="value green" id="gpu-util">-</div>
      </div>
      <div class="gauge-row">
        <div style="flex:1;">
          <div class="gauge"><div class="gauge-fill" id="gpu-util-bar" style="width:0%; background:var(--green);"></div></div>
        </div>
      </div>
      <div class="metric">
        <div class="label">Temperature</div>
        <div class="value" id="gpu-temp">-</div>
      </div>
      <div class="metric">
        <div class="label">Power</div>
        <div class="value" id="gpu-power" style="font-size:14px;">-</div>
      </div>
    </div>

    <div class="card">
      <h2>Memory</h2>
      <div class="metric">
        <div class="label">VRAM Used</div>
        <div class="value purple" id="mem-used">-</div>
      </div>
      <div class="gauge-row">
        <div style="flex:1;">
          <div class="gauge"><div class="gauge-fill" id="mem-bar" style="width:0%; background:var(--purple);"></div></div>
        </div>
      </div>
      <div class="metric">
        <div class="label">VRAM Total</div>
        <div class="value" id="mem-total" style="font-size:14px; color:var(--dim);">-</div>
      </div>
      <div class="metric">
        <div class="label">Utilization</div>
        <div class="value" id="mem-pct" style="font-size:14px;">-</div>
      </div>
    </div>

    <div class="card">
      <h2>Training Process</h2>
      <div class="metric">
        <div class="label">Phase</div>
        <div id="phase-chip">-</div>
      </div>
      <div class="metric">
        <div class="label">PID</div>
        <div class="value" id="proc-pid" style="font-size:14px;">-</div>
      </div>
      <div class="metric">
        <div class="label">CPU Usage</div>
        <div class="value" id="proc-cpu" style="font-size:14px;">-</div>
      </div>
      <div class="metric">
        <div class="label">Elapsed</div>
        <div class="value" id="proc-elapsed" style="font-size:14px;">-</div>
      </div>
    </div>
  </div>

  <!-- Loss Chart + Checkpoints -->
  <div class="grid-2">
    <div class="card">
      <h2>Loss Curve</h2>
      <canvas id="lossChart"></canvas>
      <div style="color:var(--dim); font-size:11px; margin-top:8px;">
        Latest loss: <span id="latest-loss" style="color:var(--yellow);">N/A</span>
      </div>
    </div>
    <div class="card">
      <h2>Checkpoints</h2>
      <div id="checkpoint-area" class="checkpoint-list">
        <span style="color:var(--dim);">No checkpoints yet</span>
      </div>
    </div>
  </div>

  <!-- Log Tail -->
  <div class="grid-full">
    <div class="card">
      <h2>Log Output <span class="timestamp" style="float:right;" id="log-path"></span></h2>
      <div class="log-container" id="log-output">
        <div class="log-line dim">Waiting for data...</div>
      </div>
    </div>
  </div>

<script>
// Simple canvas chart (no external deps)
class MiniChart {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.data = [];
  }
  update(losses) {
    this.data = losses;
    this.draw();
  }
  draw() {
    const c = this.ctx;
    const W = this.canvas.width = this.canvas.offsetWidth * 2;
    const H = this.canvas.height = this.canvas.offsetHeight * 2;
    c.scale(2, 2);
    const w = W / 2, h = H / 2;
    c.clearRect(0, 0, w, h);

    if (this.data.length < 2) {
      c.fillStyle = '#64748b';
      c.font = '12px monospace';
      c.textAlign = 'center';
      c.fillText('Waiting for loss data...', w/2, h/2);
      return;
    }

    const pad = { t: 10, r: 10, b: 25, l: 50 };
    const cw = w - pad.l - pad.r;
    const ch = h - pad.t - pad.b;

    const vals = this.data.map(d => d.loss);
    let minV = Math.min(...vals);
    let maxV = Math.max(...vals);
    if (maxV - minV < 0.01) { minV -= 0.1; maxV += 0.1; }
    const range = maxV - minV;

    // Grid lines
    c.strokeStyle = '#1e293b';
    c.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = pad.t + (ch * i / 4);
      c.beginPath(); c.moveTo(pad.l, y); c.lineTo(w - pad.r, y); c.stroke();
      c.fillStyle = '#64748b';
      c.font = '10px monospace';
      c.textAlign = 'right';
      c.fillText((maxV - range * i / 4).toFixed(3), pad.l - 5, y + 3);
    }

    // Loss line
    c.beginPath();
    c.strokeStyle = '#eab308';
    c.lineWidth = 1.5;
    c.shadowColor = '#eab308';
    c.shadowBlur = 4;
    for (let i = 0; i < this.data.length; i++) {
      const x = pad.l + (i / (this.data.length - 1)) * cw;
      const y = pad.t + ch - ((this.data[i].loss - minV) / range) * ch;
      if (i === 0) c.moveTo(x, y);
      else c.lineTo(x, y);
    }
    c.stroke();
    c.shadowBlur = 0;

    // X-axis label
    c.fillStyle = '#64748b';
    c.font = '10px monospace';
    c.textAlign = 'center';
    c.fillText('Training Steps', w / 2, h - 2);
  }
}

const chart = new MiniChart(document.getElementById('lossChart'));

function colorForTemp(t) {
  if (t < 60) return 'green';
  if (t < 75) return 'yellow';
  return 'red';
}

function classifyLogLine(line) {
  if (/error|traceback|exception/i.test(line)) return 'error';
  if (/warn/i.test(line)) return 'warn';
  if (/\[INFO\]/.test(line)) return 'info';
  if (/\d+%\|/.test(line)) return 'progress';
  return 'dim';
}

function updateStages(data) {
  const stages = {
    'stage-sft': { phases: ['SFT_WARMUP'], done: data.sft_done },
    'stage-dpo-m': { phases: ['DPO_METIS'], done: data.dpo_metis_done },
    'stage-dpo-r': { phases: ['DPO_RANDOM'], done: data.dpo_random_done },
    'stage-eval': { phases: ['EVALUATION'], done: data.eval_done },
  };
  for (const [id, info] of Object.entries(stages)) {
    const el = document.getElementById(id);
    el.className = 'stage';
    if (info.done) {
      el.classList.add('done');
      el.innerHTML = el.textContent.trim() + ' &#10003;';
    } else if (info.phases.includes(data.phase)) {
      el.classList.add('active');
    } else {
      el.classList.add('pending');
    }
  }
}

async function refresh() {
  try {
    const resp = await fetch('/api/status');
    const d = await resp.json();

    document.getElementById('last-update').textContent = 'Updated: ' + new Date().toLocaleTimeString();

    // GPU
    const g = d.gpu;
    if (!g.error) {
      document.getElementById('gpu-name').textContent = g.name || '-';
      document.getElementById('gpu-util').textContent = g.gpu_util >= 0 ? g.gpu_util + '%' : 'N/A';
      document.getElementById('gpu-util-bar').style.width = (g.gpu_util >= 0 ? g.gpu_util : 0) + '%';
      document.getElementById('gpu-temp').textContent = g.temp_c >= 0 ? g.temp_c + '°C' : 'N/A';
      document.getElementById('gpu-temp').className = 'value ' + colorForTemp(g.temp_c);
      document.getElementById('gpu-power').textContent =
        (g.power_w >= 0 ? g.power_w.toFixed(0) + 'W' : 'N/A') +
        (g.power_cap_w > 0 ? ' / ' + g.power_cap_w.toFixed(0) + 'W' : '');

      const memUsed = g.mem_used_mb;
      const memTotal = g.mem_total_mb;
      if (memUsed >= 0) {
        document.getElementById('mem-used').textContent = (memUsed / 1024).toFixed(1) + ' GB';
        document.getElementById('mem-total').textContent = memTotal > 0 ? (memTotal / 1024).toFixed(1) + ' GB' : 'N/A';
        const memPct = memTotal > 0 ? (memUsed / memTotal * 100) : 0;
        document.getElementById('mem-pct').textContent = memPct.toFixed(1) + '%';
        document.getElementById('mem-bar').style.width = memPct + '%';
      }
    }

    // Process
    const p = d.process;
    const statusEl = document.getElementById('proc-status');
    if (p.running) {
      statusEl.innerHTML = '<span class="dot green"></span>Training Active';
      document.getElementById('proc-pid').textContent = p.pid;
      document.getElementById('proc-cpu').textContent = p.cpu_pct + '%';
      document.getElementById('proc-elapsed').textContent = p.elapsed || '-';
    } else {
      statusEl.innerHTML = '<span class="dot red"></span>No Training Process';
      document.getElementById('proc-pid').textContent = '-';
      document.getElementById('proc-cpu').textContent = '-';
      document.getElementById('proc-elapsed').textContent = '-';
    }

    // Training
    const t = d.training;
    const phaseMap = {
      'SFT_WARMUP': ['SFT Warmup', 'sft'],
      'DPO_METIS': ['DPO METIS', 'dpo'],
      'DPO_RANDOM': ['DPO Random', 'dpo'],
      'EVALUATION': ['Evaluation', 'eval'],
      'INITIALIZING': ['Initializing', 'dpo'],
      'UNKNOWN': ['Unknown', ''],
    };
    const [phaseLabel, phaseClass] = phaseMap[t.phase] || ['Unknown', ''];
    document.getElementById('phase-chip').innerHTML =
      `<span class="chip ${phaseClass}">${phaseLabel}</span>`;

    // Progress
    document.getElementById('main-progress').style.width = t.progress_pct + '%';
    document.getElementById('main-progress-text').textContent =
      t.total_steps > 0 ? `${t.step} / ${t.total_steps}  (${t.progress_pct}%)` : '0%';
    document.getElementById('step-info').textContent = `Step: ${t.step}/${t.total_steps}`;
    document.getElementById('speed-info').textContent = `Speed: ${t.speed}`;
    document.getElementById('eta-info').textContent = `ETA: ${t.eta}`;

    updateStages(t);

    // Loss chart
    if (t.losses && t.losses.length > 0) {
      chart.update(t.losses);
      const latest = t.losses[t.losses.length - 1];
      document.getElementById('latest-loss').textContent = latest.loss.toFixed(4);
    }

    // Checkpoints
    const ckArea = document.getElementById('checkpoint-area');
    if (d.checkpoints && d.checkpoints.length > 0) {
      ckArea.innerHTML = d.checkpoints.map(ck =>
        `<div class="ckpt"><strong>${ck.name}</strong><br><span class="size">${ck.files} files, ${ck.size_mb} MB</span></div>`
      ).join('');
    }

    // Log
    const logEl = document.getElementById('log-output');
    if (t.log_lines && t.log_lines.length > 0) {
      logEl.innerHTML = t.log_lines.map(l =>
        `<div class="log-line ${classifyLogLine(l)}">${l.replace(/</g,'&lt;').replace(/>/g,'&gt;')}</div>`
      ).join('');
      logEl.scrollTop = logEl.scrollHeight;
    }

    document.getElementById('log-path').textContent = d.log_path || '';

  } catch (e) {
    document.getElementById('proc-status').innerHTML =
      '<span class="dot red"></span>Dashboard API Error';
  }
}

refresh();
setInterval(refresh, 3000);
window.addEventListener('resize', () => chart.draw());
</script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    log_path: str = str(DEFAULT_LOG)

    def log_message(self, format: str, *args: Any) -> None:
        pass  # Suppress default HTTP logs

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            self._serve_html()
        elif self.path == "/api/status":
            self._serve_status()
        else:
            self.send_error(404)

    def _serve_html(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(DASHBOARD_HTML.encode("utf-8"))

    def _serve_status(self) -> None:
        data = {
            "gpu": get_gpu_stats(),
            "process": get_process_stats(),
            "training": parse_training_log(self.log_path),
            "checkpoints": get_checkpoint_info(),
            "log_path": self.log_path,
            "timestamp": datetime.now().isoformat(),
        }
        payload = json.dumps(data, ensure_ascii=False)
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(payload.encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="METIS Training Dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Dashboard port")
    parser.add_argument("--log", type=str, default=str(DEFAULT_LOG), help="Training log file path")
    args = parser.parse_args()

    DashboardHandler.log_path = args.log

    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print(f"""
\033[36m╔══════════════════════════════════════════════╗
║   METIS Training Dashboard                   ║
║   http://localhost:{args.port}                     ║
║   Log: {Path(args.log).name:<36s}  ║
║   Auto-refresh: 3s                           ║
╚══════════════════════════════════════════════╝\033[0m
""")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
