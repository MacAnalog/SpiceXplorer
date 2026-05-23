import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#101418",
        paper: "#f7f8f5",
        circuit: "#0f766e",
        amberline: "#b45309"
      },
      boxShadow: {
        soft: "0 14px 40px rgba(16, 20, 24, 0.08)"
      }
    }
  },
  plugins: []
};

export default config;
