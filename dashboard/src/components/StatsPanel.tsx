"use client";

import type { ControllerStats, CoTStats, CognitiveSignal } from "@/lib/types";
import { PHASE_COLORS, DECISION_COLORS } from "@/lib/types";

interface StatsPanelProps {
  controller: ControllerStats | null;
  cot: CoTStats | null;
  latest: CognitiveSignal | null;
  tokenCount: number;
}

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex justify-between items-center py-0.5">
      <span className="text-ecg-muted text-xs">{label}</span>
      <span className="text-xs font-mono" style={{ color: color ?? "#e0e8f0" }}>
        {value}
      </span>
    </div>
  );
}

function ProgressBar({ value, max, color }: { value: number; max: number; color: string }) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div className="w-full h-1.5 bg-ecg-bg rounded-full overflow-hidden">
      <div
        className="h-full rounded-full transition-all duration-150"
        style={{ width: `${pct}%`, backgroundColor: color }}
      />
    </div>
  );
}

export default function StatsPanel({ controller, cot, latest, tokenCount }: StatsPanelProps) {
  const c = controller;
  const phase = latest?.cognitive_phase ?? "recall";
  const decision = latest?.decision ?? "NORMAL";

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
      {/* Controller Stats */}
      <div className="bg-ecg-panel border border-ecg-panelBorder rounded-lg p-3">
        <h3 className="text-xs font-semibold text-ecg-muted uppercase tracking-wider mb-2">
          Adaptive Controller
        </h3>
        <Stat label="Tokens" value={tokenCount.toString()} />
        <Stat label="H mean" value={c ? c.entropy_mean.toFixed(3) : "—"} />
        <Stat label="H std" value={c ? c.entropy_std.toFixed(3) : "—"} />
        <Stat label="λ (AFF)" value={c ? c.lambda_aff.toFixed(4) : "—"} />
        <Stat
          label="P(O₁)"
          value={c ? c.o1_posterior.toFixed(3) : "—"}
          color={c && c.o1_posterior > 0.5 ? "#ff4444" : undefined}
        />
        <Stat
          label="Calibrated"
          value={c?.is_calibrated ? "YES" : "NO"}
          color={c?.is_calibrated ? "#00ff88" : "#ff4444"}
        />

        <div className="mt-2">
          <div className="flex justify-between text-xs text-ecg-muted mb-0.5">
            <span>CUSUM⁺</span>
            <span>{c ? c.cusum_pos.toFixed(1) : "0"} / {c ? c.cusum_h.toFixed(1) : "4.6"}</span>
          </div>
          <ProgressBar
            value={c?.cusum_pos ?? 0}
            max={c?.cusum_h ?? 4.6}
            color={c && c.cusum_pos > c.cusum_h * 0.7 ? "#ff4444" : "#4488ff"}
          />
        </div>
      </div>

      {/* Current Signal */}
      <div className="bg-ecg-panel border border-ecg-panelBorder rounded-lg p-3">
        <h3 className="text-xs font-semibold text-ecg-muted uppercase tracking-wider mb-2">
          Live Signal
        </h3>
        <Stat
          label="Phase"
          value={phase.toUpperCase()}
          color={PHASE_COLORS[phase]}
        />
        <Stat
          label="Decision"
          value={decision}
          color={DECISION_COLORS[decision]}
        />
        <Stat label="H" value={latest ? latest.semantic_entropy.toFixed(3) : "—"} />
        <Stat label="z" value={latest ? latest.z_score.toFixed(2) : "—"} />
        <Stat label="Confidence" value={latest ? latest.confidence.toFixed(3) : "—"} />
        <Stat label="Diversity" value={latest ? latest.semantic_diversity.toFixed(3) : "—"} />
        <Stat
          label="Trend"
          value={latest?.entropy_trend ?? "—"}
          color={
            latest?.entropy_trend === "rising"
              ? "#ff8844"
              : latest?.entropy_trend === "falling"
              ? "#00ff88"
              : undefined
          }
        />
        <Stat label="Momentum" value={latest ? latest.entropy_momentum.toFixed(4) : "—"} />
      </div>

      {/* CoT & Boundary */}
      <div className="bg-ecg-panel border border-ecg-panelBorder rounded-lg p-3">
        <h3 className="text-xs font-semibold text-ecg-muted uppercase tracking-wider mb-2">
          CoT Trigger
        </h3>
        <Stat label="Injections" value={cot ? `${cot.total_injections} / ${cot.total_injections + cot.remaining_budget}` : "—"} />
        <Stat label="Budget" value={cot ? cot.remaining_budget.toString() : "—"} />
        <Stat label="Consec DEEP" value={cot ? cot.consecutive_deep.toString() : "0"} />

        <div className="mt-2">
          <div className="flex justify-between text-xs text-ecg-muted mb-0.5">
            <span>Difficulty CUSUM</span>
            <span>{cot ? cot.difficulty_cusum.toFixed(1) : "0"} / 4.0</span>
          </div>
          <ProgressBar
            value={cot?.difficulty_cusum ?? 0}
            max={4.0}
            color={cot && cot.difficulty_cusum > 2.0 ? "#ffcc00" : "#4488ff"}
          />
        </div>

        <h3 className="text-xs font-semibold text-ecg-muted uppercase tracking-wider mt-3 mb-2">
          Thresholds
        </h3>
        <Stat
          label="FAST"
          value={c ? c.fast_threshold.toFixed(2) : "1.50"}
          color="#00ff88"
        />
        <Stat
          label="DEEP"
          value={c ? c.deep_threshold.toFixed(2) : "2.00"}
          color="#ff4444"
        />
      </div>
    </div>
  );
}
