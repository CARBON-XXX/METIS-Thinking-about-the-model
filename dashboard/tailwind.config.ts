import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        ecg: {
          bg: "#0a0f1a",
          grid: "#1a2744",
          gridMajor: "#243656",
          green: "#00ff88",
          yellow: "#ffcc00",
          red: "#ff4444",
          blue: "#4488ff",
          muted: "#556688",
          panel: "#111827",
          panelBorder: "#1f2937",
        },
      },
      fontFamily: {
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
      },
      animation: {
        pulse_glow: "pulse_glow 2s ease-in-out infinite",
      },
      keyframes: {
        pulse_glow: {
          "0%, 100%": { opacity: "0.6" },
          "50%": { opacity: "1" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
