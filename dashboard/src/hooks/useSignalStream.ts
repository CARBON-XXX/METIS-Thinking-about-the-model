"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  CognitiveSignal,
  ControllerStats,
  CoTStats,
  DashboardState,
} from "@/lib/types";

/** Reward breakdown from Python reward computer */
export interface RewardEntry {
  total: number;
  coherence: number;
  calibration: number;
  phase_quality: number;
  epistemic_honesty: number;
  efficiency: number;
  promptIndex: number;
  sampleIndex: number;
  responsePreview: string;
}

/** Raw message from Python SignalBridge WebSocket */
interface BridgeSignalMessage {
  type: "signal";
  signal: CognitiveSignal;
  controller: ControllerStats;
  meta: {
    prompt_index: number;
    sample_index: number;
    total_prompts: number;
    current_prompt: string;
    phase: string;
  };
}

interface BridgeRewardMessage {
  type: "reward";
  reward: Record<string, number>;
  meta: {
    prompt_index: number;
    sample_index: number;
    total_prompts: number;
    current_prompt: string;
    response_preview: string;
  };
}

type BridgeMessage = BridgeSignalMessage | BridgeRewardMessage;

interface SignalStreamState extends DashboardState {
  promptIndex: number;
  sampleIndex: number;
  totalPrompts: number;
  currentPrompt: string;
  trainingPhase: string;
  rewardHistory: RewardEntry[];
  lastReward: RewardEntry | null;
}

const INITIAL_STATE: SignalStreamState = {
  signals: [],
  controllerStats: null,
  boundaryEvents: [],
  cotStats: null,
  connected: false,
  tokenCount: 0,
  promptIndex: 0,
  sampleIndex: 0,
  totalPrompts: 300,
  currentPrompt: "",
  trainingPhase: "waiting",
  rewardHistory: [],
  lastReward: null,
};

export function useSignalStream(url = "ws://localhost:8765"): SignalStreamState {
  const [state, setState] = useState<SignalStreamState>(INITIAL_STATE);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setState((prev) => ({ ...prev, connected: true }));
      };

      ws.onclose = () => {
        setState((prev) => ({ ...prev, connected: false }));
        // Auto-reconnect after 2s
        reconnectTimer.current = setTimeout(connect, 2000);
      };

      ws.onerror = () => {
        ws.close();
      };

      ws.onmessage = (event: MessageEvent) => {
        try {
          const msg: BridgeMessage = JSON.parse(event.data as string);

          if (msg.type === "reward") {
            const r = msg.reward;
            const entry: RewardEntry = {
              total: r.total ?? 0,
              coherence: r.coherence ?? 0,
              calibration: r.calibration ?? 0,
              phase_quality: r.phase_quality ?? 0,
              epistemic_honesty: r.epistemic_honesty ?? 0,
              efficiency: r.efficiency ?? 0,
              promptIndex: msg.meta.prompt_index,
              sampleIndex: msg.meta.sample_index,
              responsePreview: msg.meta.response_preview ?? "",
            };
            setState((prev) => ({
              ...prev,
              rewardHistory: [...prev.rewardHistory.slice(-199), entry],
              lastReward: entry,
            }));
            return;
          }

          if (msg.type !== "signal") return;

          const sig = msg.signal;
          const ctrl = msg.controller;
          const meta = msg.meta;

          setState((prev) => {
            const signals = [...prev.signals.slice(-511), sig];
            return {
              ...prev,
              signals,
              controllerStats: ctrl,
              cotStats: prev.cotStats, // CoT stats updated separately if needed
              boundaryEvents:
                sig.boundary_action !== "GENERATE"
                  ? [
                      ...prev.boundaryEvents.slice(-49),
                      {
                        state: "UNKNOWN" as const,
                        action: sig.boundary_action as "HEDGE" | "SEEK" | "REFUSE",
                        explanation: "",
                        timestamp: Date.now(),
                      },
                    ]
                  : prev.boundaryEvents,
              connected: true,
              tokenCount: prev.tokenCount + 1,
              promptIndex: meta.prompt_index,
              sampleIndex: meta.sample_index,
              totalPrompts: meta.total_prompts,
              currentPrompt: meta.current_prompt,
              trainingPhase: meta.phase,
            };
          });
        } catch {
          // Ignore malformed messages
        }
      };
    } catch {
      // Connection failed, retry
      reconnectTimer.current = setTimeout(connect, 2000);
    }
  }, [url]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return state;
}
