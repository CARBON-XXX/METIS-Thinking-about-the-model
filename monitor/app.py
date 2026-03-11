"""
METIS DPO Training Monitor — Real-time Web Dashboard
Serves training_metrics.json via API and renders Chart.js charts.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

METRICS_FILE = os.environ.get(
    "METRICS_FILE",
    str(Path(__file__).resolve().parent.parent / "experiment_output_7B_1000" / "training_metrics.json"),
)

HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>METIS DPO Training Monitor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root { --bg: #0f1117; --card: #1a1d27; --border: #2a2d3a; --text: #e0e0e0; --accent: #6c63ff; --green: #00c9a7; --red: #ff6b6b; --yellow: #ffd93d; }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background: var(--bg); color: var(--text); font-family: 'SF Mono', 'Fira Code', monospace; }
  .header { padding: 20px 32px; border-bottom: 1px solid var(--border); display:flex; align-items:center; justify-content:space-between; }
  .header h1 { font-size: 18px; color: var(--accent); }
  .status { display:flex; gap:16px; align-items:center; font-size:13px; }
  .status .dot { width:8px; height:8px; border-radius:50%; display:inline-block; }
  .dot.live { background: var(--green); animation: pulse 1.5s infinite; }
  .dot.off { background: var(--red); }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; padding:20px 32px; }
  .card { background: var(--card); border:1px solid var(--border); border-radius:10px; padding:16px; }
  .card h2 { font-size:13px; color:#888; margin-bottom:12px; text-transform:uppercase; letter-spacing:1px; }
  canvas { width:100%!important; height:280px!important; }
  .stats-row { display:grid; grid-template-columns: repeat(4,1fr); gap:12px; padding:0 32px 20px; }
  .stat { background: var(--card); border:1px solid var(--border); border-radius:8px; padding:14px; text-align:center; }
  .stat .val { font-size:22px; font-weight:700; color: var(--accent); }
  .stat .label { font-size:11px; color:#666; margin-top:4px; }
  .gpu-bar { height:6px; background:#333; border-radius:3px; margin-top:6px; overflow:hidden; }
  .gpu-bar .fill { height:100%; border-radius:3px; transition: width 0.5s; }
  @media(max-width:900px){ .grid{grid-template-columns:1fr;} .stats-row{grid-template-columns:repeat(2,1fr);} }
</style>
</head>
<body>
<div class="header">
  <h1>⚡ METIS DPO Training Monitor</h1>
  <div class="status">
    <span>Auto-refresh: 10s</span>
    <span class="dot live" id="liveDot"></span>
    <span id="statusText">Connecting...</span>
  </div>
</div>

<div class="stats-row" id="statsRow">
  <div class="stat"><div class="val" id="stepVal">-</div><div class="label">Step / Total</div></div>
  <div class="stat"><div class="val" id="epochVal">-</div><div class="label">Epoch</div></div>
  <div class="stat"><div class="val" id="gpuVal">-</div><div class="label">GPU Memory</div>
    <div class="gpu-bar"><div class="fill" id="gpuFill" style="width:0%;background:var(--green)"></div></div>
  </div>
  <div class="stat"><div class="val" id="etaVal">-</div><div class="label">ETA</div></div>
</div>

<div class="grid">
  <div class="card">
    <h2>Rewards / Margins (convergence)</h2>
    <canvas id="marginsChart"></canvas>
  </div>
  <div class="card">
    <h2>Rewards / Accuracies (climb rate)</h2>
    <canvas id="accChart"></canvas>
  </div>
  <div class="card">
    <h2>Training Loss</h2>
    <canvas id="lossChart"></canvas>
  </div>
  <div class="card">
    <h2>Rewards / Chosen vs Rejected</h2>
    <canvas id="rewardsChart"></canvas>
  </div>
</div>

<script>
const REFRESH_MS = 10000;
const chartOpts = (title) => ({
  responsive: true,
  animation: { duration: 400 },
  plugins: { legend: { labels: { color:'#888', font:{size:11} } } },
  scales: {
    x: { title:{display:true, text:'Step', color:'#666'}, ticks:{color:'#555'}, grid:{color:'#222'} },
    y: { title:{display:true, text:title, color:'#666'}, ticks:{color:'#555'}, grid:{color:'#222'} }
  }
});

function mkChart(id, label, color, yTitle) {
  return new Chart(document.getElementById(id), {
    type: 'line',
    data: { labels:[], datasets:[
      { label: 'METIS '+label, data:[], borderColor:color, backgroundColor:color+'33', tension:0.3, pointRadius:1, borderWidth:2 },
      { label: 'Random '+label, data:[], borderColor:'#ff6b6b', backgroundColor:'#ff6b6b33', tension:0.3, pointRadius:1, borderWidth:2 }
    ]},
    options: chartOpts(yTitle)
  });
}

const marginsChart = mkChart('marginsChart', 'Margins', '#6c63ff', 'Margin');
const accChart     = mkChart('accChart', 'Accuracy', '#00c9a7', 'Accuracy');
const lossChart    = mkChart('lossChart', 'Loss', '#ffd93d', 'Loss');
const rewardsChart = new Chart(document.getElementById('rewardsChart'), {
  type: 'line',
  data: { labels:[], datasets:[
    { label:'Chosen', data:[], borderColor:'#00c9a7', tension:0.3, pointRadius:1, borderWidth:2 },
    { label:'Rejected', data:[], borderColor:'#ff6b6b', tension:0.3, pointRadius:1, borderWidth:2 }
  ]},
  options: chartOpts('Reward')
});

function extract(entries, key) { return entries.map(e => e[key] ?? null); }
function steps(entries) { return entries.map(e => e.step); }

async function refresh() {
  try {
    const r = await fetch('/api/metrics');
    const d = await r.json();
    const m = d.metis || [];
    const rnd = d.random || [];

    // Margins
    marginsChart.data.labels = steps(m);
    marginsChart.data.datasets[0].data = extract(m, 'rewards/margins');
    marginsChart.data.datasets[1].data = extract(rnd, 'rewards/margins');
    marginsChart.update('none');

    // Accuracies
    accChart.data.labels = steps(m);
    accChart.data.datasets[0].data = extract(m, 'rewards/accuracies');
    accChart.data.datasets[1].data = extract(rnd, 'rewards/accuracies');
    accChart.update('none');

    // Loss
    lossChart.data.labels = steps(m);
    lossChart.data.datasets[0].data = extract(m, 'loss');
    lossChart.data.datasets[1].data = extract(rnd, 'loss');
    lossChart.update('none');

    // Chosen vs Rejected (METIS group only)
    rewardsChart.data.labels = steps(m);
    rewardsChart.data.datasets[0].data = extract(m, 'rewards/chosen');
    rewardsChart.data.datasets[1].data = extract(m, 'rewards/rejected');
    rewardsChart.update('none');

    // Stats
    if (m.length > 0) {
      const last = m[m.length-1];
      document.getElementById('stepVal').textContent = last.step + ' / ' + (d.total_steps || '?');
      document.getElementById('epochVal').textContent = (last.epoch || 0).toFixed(2);
      document.getElementById('statusText').textContent = 'Training';
      document.getElementById('liveDot').className = 'dot live';
    }

    // GPU
    const g = await fetch('/api/gpu');
    const gd = await g.json();
    if (gd.used_gb !== undefined) {
      const pct = Math.round(gd.used_gb / gd.total_gb * 100);
      document.getElementById('gpuVal').textContent = gd.used_gb.toFixed(1) + ' / ' + gd.total_gb.toFixed(0) + ' GB';
      document.getElementById('gpuFill').style.width = pct + '%';
      document.getElementById('gpuFill').style.background = pct > 95 ? '#ff6b6b' : pct > 80 ? '#ffd93d' : '#00c9a7';
    }

    // ETA
    if (m.length >= 2) {
      const totalSteps = d.total_steps || 192;
      const elapsed = m[m.length-1].timestamp - m[0].timestamp;
      const secPerStep = elapsed / (m.length - 1);
      const remaining = (totalSteps - m[m.length-1].step) * secPerStep;
      const hrs = Math.floor(remaining/3600);
      const mins = Math.floor((remaining%3600)/60);
      document.getElementById('etaVal').textContent = hrs + 'h ' + mins + 'm';
    }

  } catch(e) {
    document.getElementById('statusText').textContent = 'Waiting for data...';
    document.getElementById('liveDot').className = 'dot off';
  }
}

refresh();
setInterval(refresh, REFRESH_MS);
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/metrics")
def api_metrics():
    if not os.path.exists(METRICS_FILE):
        return jsonify({"metis": [], "random": [], "total_steps": 0})
    with open(METRICS_FILE, "r") as f:
        data = json.load(f)
    # Estimate total steps from METIS group epoch info
    metis = data.get("metis", [])
    total_steps = 192  # fallback
    if metis:
        last = metis[-1]
        if last.get("epoch") and last["epoch"] > 0:
            total_steps = max(total_steps, int(last["step"] / last["epoch"] * 3))
    data["total_steps"] = total_steps
    return jsonify(data)


@app.route("/api/gpu")
def api_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
            text=True,
        ).strip()
        used, total = [float(x) for x in out.split(",")]
        return jsonify({"used_gb": used / 1024, "total_gb": total / 1024})
    except Exception:
        return jsonify({"used_gb": 0, "total_gb": 128})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=False)
