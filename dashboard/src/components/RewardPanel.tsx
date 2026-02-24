"use client";

import type { RewardEntry } from "@/hooks/useSignalStream";

const REWARD_COLORS: Record<string, string> = {
  coherence: "#4ade80",
  calibration: "#60a5fa",
  phase_quality: "#c084fc",
  epistemic_honesty: "#facc15",
  efficiency: "#f97316",
};

const REWARD_LABELS: Record<string, string> = {
  coherence: "R_coh",
  calibration: "R_cal",
  phase_quality: "R_phase",
  epistemic_honesty: "R_epist",
  efficiency: "R_eff",
};

function Bar({ value, color, label }: { value: number; color: string; label: string }) {
  const pct = Math.max(0, Math.min(100, (value + 1) * 50));
  const isPositive = value >= 0;
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-14 text-ecg-muted font-mono">{label}</span>
      <div className="flex-1 h-3 bg-ecg-bg rounded-full overflow-hidden relative">
        <div className="absolute inset-y-0 left-1/2 w-px bg-ecg-panelBorder" />
        {isPositive ? (
          <div
            className="absolute inset-y-0 rounded-r-full transition-all duration-300"
            style={{
              left: "50%",
              width: `${(value / 1) * 50}%`,
              backgroundColor: color,
              opacity: 0.8,
            }}
          />
        ) : (
          <div
            className="absolute inset-y-0 rounded-l-full transition-all duration-300"
            style={{
              right: "50%",
              width: `${(Math.abs(value) / 1) * 50}%`,
              backgroundColor: "#ef4444",
              opacity: 0.8,
            }}
          />
        )}
      </div>
      <span
        className="w-12 text-right font-mono"
        style={{ color: isPositive ? color : "#ef4444" }}
      >
        {value >= 0 ? "+" : ""}
        {value.toFixed(3)}
      </span>
    </div>
  );
}

function MiniSparkline({ data, color }: { data: number[]; color: string }) {
  if (data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const w = 120;
  const h = 24;
  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - ((v - min) / range) * h;
    return `${x},${y}`;
  });
  return (
    <svg width={w} height={h} className="opacity-60">
      <polyline
        points={points.join(" ")}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
      />
    </svg>
  );
}

export default function RewardPanel({
  lastReward,
  rewardHistory,
}: {
  lastReward: RewardEntry | null;
  rewardHistory: RewardEntry[];
}) {
  if (!lastReward && rewardHistory.length === 0) {
    return (
      <section className="bg-ecg-panel border border-ecg-panelBorder rounded-lg p-3">
        <h2 className="text-xs font-semibold text-ecg-muted uppercase tracking-wider mb-2">
          Reward Breakdown
        </h2>
        <p className="text-xs text-ecg-muted opacity-50">Waiting for first sample...</p>
      </section>
    );
  }

  const r = lastReward ?? rewardHistory[rewardHistory.length - 1];
  const totalHistory = rewardHistory.map((e) => e.total);
  const avgTotal =
    rewardHistory.length > 0
      ? rewardHistory.reduce((s, e) => s + e.total, 0) / rewardHistory.length
      : 0;

  const components: [string, number][] = [
    ["coherence", r.coherence],
    ["calibration", r.calibration],
    ["phase_quality", r.phase_quality],
    ["epistemic_honesty", r.epistemic_honesty],
    ["efficiency", r.efficiency],
  ];

  return (
    <section className="bg-ecg-panel border border-ecg-panelBorder rounded-lg p-3">
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-xs font-semibold text-ecg-muted uppercase tracking-wider">
          Reward Breakdown
        </h2>
        <div className="flex items-center gap-3 text-xs">
          <span className="text-ecg-muted">
            Total:{" "}
            <span
              className="font-mono font-bold"
              style={{ color: r.total >= 0 ? "#4ade80" : "#ef4444" }}
            >
              {r.total >= 0 ? "+" : ""}
              {r.total.toFixed(4)}
            </span>
          </span>
          <span className="text-ecg-muted">
            Avg:{" "}
            <span className="font-mono text-white">
              {avgTotal >= 0 ? "+" : ""}
              {avgTotal.toFixed(4)}
            </span>
          </span>
          <span className="text-ecg-muted">
            n={rewardHistory.length}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-x-4 gap-y-1">
        <div className="space-y-1">
          {components.map(([key, val]) => (
            <Bar
              key={key}
              value={val}
              color={REWARD_COLORS[key] ?? "#888"}
              label={REWARD_LABELS[key] ?? key}
            />
          ))}
        </div>
        <div className="flex flex-col items-center justify-center">
          <span className="text-xs text-ecg-muted mb-1">R_total trend</span>
          <MiniSparkline data={totalHistory.slice(-60)} color="#4ade80" />
          {r.responsePreview && (
            <p className="text-xs text-ecg-muted mt-2 truncate max-w-[200px] opacity-50">
              &quot;{r.responsePreview}&quot;
            </p>
          )}
        </div>
      </div>
    </section>
  );
}
