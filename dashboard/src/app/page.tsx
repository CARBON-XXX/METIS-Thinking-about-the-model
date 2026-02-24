"use client";

import { useSignalStream } from "@/hooks/useSignalStream";
import ECGCanvas from "@/components/ECGCanvas";
import StatsPanel from "@/components/StatsPanel";
import RewardPanel from "@/components/RewardPanel";
import { PHASE_COLORS, DECISION_COLORS } from "@/lib/types";

/** Decision history timeline (last 60 decisions) */
function DecisionTimeline({ signals }: { signals: { decision: "FAST" | "NORMAL" | "DEEP" }[] }) {
  const recent = signals.slice(-60);
  return (
    <div className="flex items-center gap-px h-4 overflow-hidden">
      {recent.map((s, i) => (
        <div
          key={i}
          className="w-1.5 rounded-sm transition-all duration-75"
          style={{
            height: s.decision === "DEEP" ? 16 : s.decision === "FAST" ? 6 : 10,
            backgroundColor: DECISION_COLORS[s.decision],
            opacity: 0.4 + (i / recent.length) * 0.6,
          }}
        />
      ))}
    </div>
  );
}

/** Header with connection status and phase badge */
function Header({
  phase,
  tokenCount,
  connected,
}: {
  phase: string;
  tokenCount: number;
  connected: boolean;
}) {
  const phaseKey = phase as keyof typeof PHASE_COLORS;
  return (
    <header className="flex items-center justify-between px-4 py-3 border-b border-ecg-panelBorder">
      <div className="flex items-center gap-3">
        <h1 className="text-base font-bold tracking-tight">
          <span className="text-ecg-green">METIS</span>
          <span className="text-ecg-muted ml-1.5 font-normal text-sm">Cognitive ECG</span>
        </h1>
        <div
          className="px-2 py-0.5 rounded text-xs font-semibold uppercase tracking-wider"
          style={{
            color: PHASE_COLORS[phaseKey] ?? "#556688",
            backgroundColor: `${PHASE_COLORS[phaseKey] ?? "#556688"}15`,
            border: `1px solid ${PHASE_COLORS[phaseKey] ?? "#556688"}40`,
          }}
        >
          {phase}
        </div>
      </div>
      <div className="flex items-center gap-4 text-xs text-ecg-muted">
        <span>Token #{tokenCount}</span>
        <div className="flex items-center gap-1.5">
          <div
            className={`w-2 h-2 rounded-full ${
              connected ? "bg-ecg-green animate-pulse_glow" : "bg-red-500"
            }`}
          />
          <span>{connected ? "LIVE" : "OFFLINE"}</span>
        </div>
      </div>
    </header>
  );
}

export default function DashboardPage() {
  const {
    signals,
    controllerStats,
    cotStats,
    connected,
    tokenCount,
    promptIndex,
    sampleIndex,
    totalPrompts,
    currentPrompt,
    trainingPhase,
    rewardHistory,
    lastReward,
  } = useSignalStream("ws://localhost:8765");

  const latest = signals.length > 0 ? signals[signals.length - 1] : null;
  const phase = latest?.cognitive_phase ?? "recall";
  const fastT = controllerStats?.fast_threshold ?? 1.5;
  const deepT = controllerStats?.deep_threshold ?? 2.0;
  const progress = totalPrompts > 0 ? (promptIndex / totalPrompts) * 100 : 0;

  return (
    <div className="min-h-screen flex flex-col bg-ecg-bg">
      <Header phase={phase} tokenCount={tokenCount} connected={connected} />

      <main className="flex-1 p-4 space-y-3 max-w-7xl mx-auto w-full">
        {/* Training Progress */}
        <section className="bg-ecg-panel border border-ecg-panelBorder rounded-lg p-3">
          <div className="flex items-center justify-between mb-1.5">
            <h2 className="text-xs font-semibold text-ecg-muted uppercase tracking-wider">
              Training — {trainingPhase.toUpperCase()}
            </h2>
            <span className="text-xs text-ecg-muted">
              Prompt {promptIndex}/{totalPrompts} · Sample {sampleIndex}
            </span>
          </div>
          <div className="w-full h-1.5 bg-ecg-bg rounded-full overflow-hidden mb-1.5">
            <div
              className="h-full rounded-full bg-ecg-green transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-xs text-ecg-muted truncate opacity-70">
            {currentPrompt || (connected ? "Waiting for signal..." : "Not connected — start training with bridge")}
          </p>
        </section>

        {/* ECG Waveform */}
        <section>
          <div className="flex items-center justify-between mb-1.5">
            <h2 className="text-xs font-semibold text-ecg-muted uppercase tracking-wider">
              Entropy Waveform
            </h2>
            <div className="flex items-center gap-3 text-xs text-ecg-muted">
              <span>
                H = <span className="text-white font-mono">{latest?.semantic_entropy.toFixed(2) ?? "—"}</span>
              </span>
              <span>
                z = <span className="text-white font-mono">{latest?.z_score.toFixed(2) ?? "—"}</span>
              </span>
            </div>
          </div>
          <ECGCanvas signals={signals} fastThreshold={fastT} deepThreshold={deepT} />
        </section>

        {/* Decision Timeline */}
        <section className="bg-ecg-panel border border-ecg-panelBorder rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-xs font-semibold text-ecg-muted uppercase tracking-wider">
              Decision Stream
            </h2>
            <div className="flex items-center gap-3 text-xs">
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-ecg-green" /> FAST
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-ecg-blue" /> NORMAL
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-ecg-red" /> DEEP
              </span>
            </div>
          </div>
          <DecisionTimeline signals={signals} />
        </section>

        {/* Reward Breakdown */}
        <RewardPanel lastReward={lastReward} rewardHistory={rewardHistory} />

        {/* Stats Panels */}
        <StatsPanel
          controller={controllerStats}
          cot={cotStats}
          latest={latest}
          tokenCount={tokenCount}
        />

        {/* Footer */}
        <footer className="text-center text-xs text-ecg-muted py-2 opacity-50">
          METIS Metacognitive Operating System — WebGPU Real-time Dashboard v0.1
        </footer>
      </main>
    </div>
  );
}
