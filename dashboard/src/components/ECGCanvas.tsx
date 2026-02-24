"use client";

import { useEffect, useRef, useState } from "react";
import { ECGRenderer } from "@/lib/wgpu/ecg-renderer";
import type { CognitiveSignal } from "@/lib/types";

interface ECGCanvasProps {
  signals: CognitiveSignal[];
  fastThreshold: number;
  deepThreshold: number;
}

const DECISION_INT: Record<CognitiveSignal["decision"], number> = {
  FAST: 0,
  NORMAL: 1,
  DEEP: 2,
};

export default function ECGCanvas({
  signals,
  fastThreshold,
  deepThreshold,
}: ECGCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<ECGRenderer | null>(null);
  const [gpuReady, setGpuReady] = useState(false);
  const [fallback, setFallback] = useState(false);

  // Initialize WebGPU renderer
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const renderer = new ECGRenderer();
    rendererRef.current = renderer;

    const initGPU = async () => {
      const ok = await renderer.init(canvas);
      if (ok) {
        setGpuReady(true);
        renderer.startLoop();
      } else {
        setFallback(true);
      }
    };

    initGPU();

    return () => {
      renderer.dispose();
      rendererRef.current = null;
    };
  }, []);

  // Resize handler
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.floor(width * dpr);
        canvas.height = Math.floor(height * dpr);
      }
    });

    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  // Push signal data to GPU
  useEffect(() => {
    const renderer = rendererRef.current;
    if (!renderer || !gpuReady || signals.length === 0) return;

    const latest = signals[signals.length - 1];
    renderer.pushSample(latest.semantic_entropy, DECISION_INT[latest.decision]);
    renderer.setThresholds(fastThreshold, deepThreshold);
  }, [signals, gpuReady, fastThreshold, deepThreshold]);

  // Canvas 2D fallback for browsers without WebGPU
  useEffect(() => {
    if (!fallback) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const draw = () => {
      const w = canvas.width;
      const h = canvas.height;
      ctx.fillStyle = "#0a0f1a";
      ctx.fillRect(0, 0, w, h);

      if (signals.length < 2) return;

      const margin = { l: 40, r: 20, t: 20, b: 20 };
      const pw = w - margin.l - margin.r;
      const ph = h - margin.t - margin.b;

      // Grid
      ctx.strokeStyle = "#1a2744";
      ctx.lineWidth = 0.5;
      for (let i = 0; i <= 10; i++) {
        const y = margin.t + (ph * i) / 10;
        ctx.beginPath();
        ctx.moveTo(margin.l, y);
        ctx.lineTo(w - margin.r, y);
        ctx.stroke();
      }

      // Waveform
      ctx.beginPath();
      ctx.strokeStyle = "#00ff88";
      ctx.lineWidth = 2;
      ctx.shadowColor = "#00ff88";
      ctx.shadowBlur = 8;

      const n = signals.length;
      for (let i = 0; i < n; i++) {
        const x = margin.l + (i / (n - 1)) * pw;
        const e = signals[i].semantic_entropy;
        const yNorm = Math.min(e / 5, 1);
        const y = margin.t + ph - yNorm * ph;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;
    };

    draw();
  }, [signals, fallback]);

  return (
    <div className="ecg-canvas-wrap w-full" style={{ height: 280 }}>
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ display: "block" }}
      />
      {!gpuReady && !fallback && (
        <div className="absolute inset-0 flex items-center justify-center text-ecg-muted text-sm">
          Initializing WebGPU...
        </div>
      )}
      {fallback && (
        <div className="absolute top-2 right-3 text-xs text-ecg-muted opacity-60">
          Canvas2D fallback
        </div>
      )}
    </div>
  );
}
