"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { CognitiveSignal, ControllerStats, CoTStats } from "@/lib/types";

/**
 * Generates realistic mock METIS cognitive signals for demo/development.
 * Simulates entropy fluctuations, phase transitions, and decision patterns.
 */

interface MockState {
  signals: CognitiveSignal[];
  latest: CognitiveSignal | null;
  controllerStats: ControllerStats;
  cotStats: CoTStats;
  tokenCount: number;
}

const DECISION_MAP: CognitiveSignal["decision"][] = ["FAST", "NORMAL", "DEEP"];
const PHASE_MAP: CognitiveSignal["cognitive_phase"][] = [
  "fluent",
  "recall",
  "reasoning",
  "exploration",
  "confusion",
];

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function generateSignal(t: number, prev: CognitiveSignal | null): CognitiveSignal {
  // Base entropy with multi-frequency oscillation (simulates real model behavior)
  const base =
    1.2 +
    0.6 * Math.sin(t * 0.03) +
    0.4 * Math.sin(t * 0.11 + 1.5) +
    0.3 * Math.sin(t * 0.23 + 3.0);

  // Random walk component
  const prevE = prev?.semantic_entropy ?? base;
  const walk = (Math.random() - 0.5) * 0.3;
  const entropy = clamp(prevE * 0.85 + base * 0.15 + walk, 0.1, 4.5);

  const tokenEntropy = entropy * 0.9 + Math.random() * 0.2;
  const diversity = clamp(0.3 + entropy * 0.12 + (Math.random() - 0.5) * 0.15, 0.0, 1.0);
  const confidence = clamp(1.0 - entropy * 0.22 + (Math.random() - 0.5) * 0.1, 0.05, 0.99);

  // z-score simulation
  const mean = 1.5;
  const std = 0.5;
  const z = (entropy - mean) / std;

  // Decision based on entropy
  let decision: CognitiveSignal["decision"];
  if (entropy < 1.2) decision = "FAST";
  else if (entropy > 2.5) decision = "DEEP";
  else decision = "NORMAL";

  // Phase based on entropy + diversity
  let phase: CognitiveSignal["cognitive_phase"];
  if (entropy < 0.8) phase = "fluent";
  else if (entropy < 1.3 && confidence > 0.7) phase = "recall";
  else if (entropy > 2.0 && diversity > 0.6) phase = "exploration";
  else if (entropy > 2.5 && diversity < 0.5) phase = "confusion";
  else phase = "reasoning";

  // Trend
  const prevEntropy = prev?.semantic_entropy ?? entropy;
  const diff = entropy - prevEntropy;
  let trend: CognitiveSignal["entropy_trend"];
  if (Math.abs(diff) < 0.1) trend = "stable";
  else if (diff > 0) trend = "rising";
  else trend = "falling";

  const momentum = prev ? diff * 0.3 + (prev.entropy_momentum ?? 0) * 0.7 : 0;

  return {
    semantic_entropy: entropy,
    token_entropy: tokenEntropy,
    semantic_diversity: diversity,
    confidence,
    z_score: z,
    decision,
    entropy_trend: trend,
    cognitive_phase: phase,
    entropy_momentum: momentum,
  };
}

export function useMockSignals(intervalMs = 50): MockState {
  const [state, setState] = useState<MockState>({
    signals: [],
    latest: null,
    controllerStats: {
      entropy_mean: 1.5,
      entropy_std: 0.5,
      entropy_skew: 0,
      entropy_kurt: 0,
      fast_threshold: 1.2,
      deep_threshold: 2.5,
      lambda_aff: 0.995,
      o1_posterior: 0.1,
      cusum_pos: 0,
      cusum_neg: 0,
      cusum_h: 4.6,
      change_detected: 0,
      is_calibrated: 0,
      step_count: 0,
    },
    cotStats: {
      total_injections: 0,
      difficulty_cusum: 0,
      momentum_acc: 0,
      consecutive_deep: 0,
      remaining_budget: 3,
    },
    tokenCount: 0,
  });

  const tickRef = useRef(0);
  const prevRef = useRef<CognitiveSignal | null>(null);
  const cusumRef = useRef(0);
  const cotCusumRef = useRef(0);

  useEffect(() => {
    const timer = setInterval(() => {
      tickRef.current++;
      const t = tickRef.current;
      const sig = generateSignal(t, prevRef.current);
      prevRef.current = sig;

      // Simulate CUSUM accumulation
      if (sig.z_score > 0.5) {
        cusumRef.current += (sig.z_score - 0.3) * sig.semantic_diversity;
      } else if (sig.z_score < 0) {
        cusumRef.current *= 0.92;
      }
      cusumRef.current = Math.max(0, cusumRef.current);

      // Simulate CoT CUSUM
      if (sig.decision === "DEEP") {
        cotCusumRef.current += 0.5;
      } else if (sig.z_score < 0) {
        cotCusumRef.current *= 0.9;
      }
      cotCusumRef.current = Math.max(0, cotCusumRef.current);

      setState((prev) => {
        const signals = [...prev.signals.slice(-511), sig];
        return {
          signals,
          latest: sig,
          controllerStats: {
            ...prev.controllerStats,
            entropy_mean: prev.controllerStats.entropy_mean * 0.99 + sig.semantic_entropy * 0.01,
            entropy_std: 0.5 + Math.random() * 0.1,
            fast_threshold: 1.0 + Math.sin(t * 0.005) * 0.3,
            deep_threshold: 2.2 + Math.sin(t * 0.005) * 0.3,
            cusum_pos: cusumRef.current,
            lambda_aff: 0.99 - Math.abs(sig.z_score) * 0.002,
            o1_posterior: clamp(0.1 + cusumRef.current * 0.05, 0, 1),
            cusum_h: 4.6,
            change_detected: cusumRef.current > 4.6 ? 1 : 0,
            is_calibrated: t > 20 ? 1 : 0,
            step_count: t,
          },
          cotStats: {
            total_injections: Math.floor(t / 200),
            difficulty_cusum: cotCusumRef.current,
            momentum_acc: sig.entropy_momentum,
            consecutive_deep: sig.decision === "DEEP" ? (prev.cotStats.consecutive_deep + 1) : 0,
            remaining_budget: Math.max(0, 3 - Math.floor(t / 200)),
          },
          tokenCount: t,
        };
      });
    }, intervalMs);

    return () => clearInterval(timer);
  }, [intervalMs]);

  return state;
}
