import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {},
      boxShadow: {
        soft: "0 14px 40px rgba(16, 20, 24, 0.08)"
      }
    }
  },
  plugins: []
};

export default config;
