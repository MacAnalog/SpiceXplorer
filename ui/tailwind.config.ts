import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Surface
        bg: "#fafafa",
        panel: "#ffffff",
        hairline: "#f4f4f5",
        border: "#e4e4e7",
        // Ink
        fg: "#0a0a0a",
        muted: "#71717a",
        faint: "#a1a1aa",
        // Brand
        primary: "#4f46e5",
        "primary-soft": "#eef2ff",
        secondary: "#0891b2",
        "secondary-soft": "#ecfeff",
        tertiary: "#ea580c",
        // Status
        ok: "#059669",
        "ok-soft": "#d1fae5",
        danger: "#dc2626",
        "danger-soft": "#fef2f2",
        "warn-soft": "#fef3c7",
      },
      fontFamily: {
        sans: ["var(--font-sans)", "Inter Tight", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "IBM Plex Mono", "ui-monospace", "monospace"],
      },
      boxShadow: {
        soft: "0 14px 40px rgba(16, 20, 24, 0.08)",
        seg: "0 1px 2px rgba(0,0,0,0.06)",
      },
    },
  },
  plugins: [],
};

export default config;
