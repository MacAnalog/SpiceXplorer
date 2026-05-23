"use client";
import { useEffect, useState } from "react";
import { BarChart3, CircleDot, Play, SlidersHorizontal, Zap } from "lucide-react";
import { cn } from "@/lib/utils";
import { useProjectStore } from "@/stores/projectStore";
import { useRunStore } from "@/stores/runStore";
import { api } from "@/lib/api";
import { SetupTab } from "@/components/tabs/SetupTab";
import { ScoreShapingTab } from "@/components/tabs/ScoreShapingTab";
import { OptimizeTab } from "@/components/tabs/OptimizeTab";
import { ExplorerTab } from "@/components/tabs/ExplorerTab";
import type { DemoConfig } from "@/types/api";

type TabId = "setup" | "shaping" | "optimize" | "explorer";

const TABS: Array<{ id: TabId; label: string; icon: typeof Zap }> = [
  { id: "setup", label: "Setup", icon: Zap },
  { id: "shaping", label: "Score Shaping", icon: SlidersHorizontal },
  { id: "optimize", label: "Optimize", icon: Play },
  { id: "explorer", label: "Explorer", icon: BarChart3 },
];

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabId>("setup");
  const [demoConfig, setDemoConfig] = useState<DemoConfig | null>(null);
  const { isApplied } = useProjectStore();
  const { isRunning } = useRunStore();

  useEffect(() => {
    api.config().then(setDemoConfig).catch(() => null);
  }, []);

  return (
    <main className="min-h-screen bg-zinc-50">
      <header className="border-b border-zinc-200 bg-white px-4 py-3">
        <div className="mx-auto flex max-w-7xl items-center justify-between">
          <div className="flex items-center gap-2.5">
            <Zap className="h-5 w-5 text-indigo-600" />
            <span className="text-lg font-semibold text-zinc-900">SpiceXplorer</span>
            <span className="hidden rounded-full border border-zinc-200 px-2 py-0.5 text-[11px] text-zinc-400 sm:inline">
              NEWCAS 2026
            </span>
          </div>
          {isRunning && (
            <div className="flex items-center gap-1.5 rounded-full bg-indigo-50 px-3 py-1 text-xs font-medium text-indigo-700">
              <CircleDot className="h-3 w-3 animate-pulse text-indigo-600" />
              Running
            </div>
          )}
        </div>
      </header>

      <div className="mx-auto max-w-7xl px-4 py-4">
        {/* Tab nav */}
        <nav className="mb-4 flex gap-0 border-b border-zinc-200">
          {TABS.map((tab) => {
            const Icon = tab.icon;
            const disabled = !isApplied && tab.id !== "setup" && tab.id !== "explorer";
            return (
              <button
                key={tab.id}
                onClick={() => !disabled && setActiveTab(tab.id)}
                disabled={disabled}
                className={cn(
                  "inline-flex items-center gap-1.5 border-b-2 px-4 pb-3 pt-1 text-sm font-medium transition -mb-px",
                  activeTab === tab.id
                    ? "border-indigo-600 text-indigo-700"
                    : disabled
                      ? "cursor-not-allowed border-transparent text-zinc-300"
                      : "border-transparent text-zinc-600 hover:border-zinc-300 hover:text-zinc-800",
                )}
              >
                <Icon className="h-4 w-4" />
                {tab.label}
              </button>
            );
          })}
        </nav>

        {/* Tab content */}
        <section>
          {activeTab === "setup" && <SetupTab demoConfig={demoConfig} />}
          {activeTab === "shaping" && <ScoreShapingTab />}
          {activeTab === "optimize" && <OptimizeTab demoConfig={demoConfig} />}
          {activeTab === "explorer" && <ExplorerTab />}
        </section>
      </div>
    </main>
  );
}
