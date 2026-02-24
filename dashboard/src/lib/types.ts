/**
 * METIS Cognitive Signal — mirrors Python CognitiveSignal dataclass.
 * All fields required; no `any` types.
 */
export interface CognitiveSignal {
  semantic_entropy: number;
  token_entropy: number;
  semantic_diversity: number;
  confidence: number;
  z_score: number;
  decision: "FAST" | "NORMAL" | "DEEP";
  entropy_trend: "stable" | "rising" | "falling" | "oscillating";
  cognitive_phase: "fluent" | "recall" | "reasoning" | "exploration" | "confusion";
  entropy_momentum: number;
  boundary_action?: "GENERATE" | "HEDGE" | "SEEK" | "REFUSE";
  token_surprise?: number;
  cusum_alarm?: boolean;
}

/** Controller stats from AdaptiveController.stats property */
export interface ControllerStats {
  entropy_mean: number;
  entropy_std: number;
  entropy_skew: number;
  entropy_kurt: number;
  fast_threshold: number;
  deep_threshold: number;
  lambda_aff: number;
  o1_posterior: number;
  cusum_pos: number;
  cusum_neg: number;
  cusum_h: number;
  change_detected: number;
  is_calibrated: number;
  step_count: number;
}

/** Boundary guard action event */
export interface BoundaryEvent {
  state: "KNOWN" | "LIKELY" | "UNCERTAIN" | "UNKNOWN";
  action: "GENERATE" | "HEDGE" | "SEEK" | "REFUSE";
  explanation: string;
  timestamp: number;
}

/** CoT manager stats */
export interface CoTStats {
  total_injections: number;
  difficulty_cusum: number;
  momentum_acc: number;
  consecutive_deep: number;
  remaining_budget: number;
}

/** Full dashboard state snapshot */
export interface DashboardState {
  signals: CognitiveSignal[];
  controllerStats: ControllerStats | null;
  boundaryEvents: BoundaryEvent[];
  cotStats: CoTStats | null;
  connected: boolean;
  tokenCount: number;
}

/** Decision color mapping */
export const DECISION_COLORS: Record<CognitiveSignal["decision"], string> = {
  FAST: "#00ff88",
  NORMAL: "#4488ff",
  DEEP: "#ff4444",
};

/** Phase color mapping */
export const PHASE_COLORS: Record<CognitiveSignal["cognitive_phase"], string> = {
  fluent: "#00ff88",
  recall: "#44ddff",
  reasoning: "#ffcc00",
  exploration: "#ff8844",
  confusion: "#ff4444",
};
