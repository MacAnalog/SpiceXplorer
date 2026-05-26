"use client";
import { useEffect, useState } from "react";
import { useProjectStore } from "@/stores/projectStore";
import { useExplorerStore } from "@/stores/explorerStore";
import { api } from "@/lib/api";
import { SetupTab } from "@/components/tabs/SetupTab";
import { SchematicTab } from "@/components/tabs/SchematicTab";
import { ScoreShapingTab } from "@/components/tabs/ScoreShapingTab";
import { OptimizeTab } from "@/components/tabs/OptimizeTab";
import { ExplorerTab } from "@/components/tabs/ExplorerTab";
import { HealthTab } from "@/components/tabs/HealthTab";
import { Topbar, type TabId } from "@/components/shell/Topbar";
import { LeftRail } from "@/components/shell/LeftRail";
import type { AppConfig } from "@/types/api";

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabId>("setup");
  const [appConfig, setAppConfig] = useState<AppConfig | null>(null);
  const { isApplied } = useProjectStore();
  const { setAvailableCheckpoints } = useExplorerStore();

  useEffect(() => {
    api.config().then(setAppConfig).catch(() => null);
    api.listCheckpoints().then(setAvailableCheckpoints).catch(() => null);
  }, [setAvailableCheckpoints]);

  // Keyboard shortcuts: 1/2/3/4
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // ignore typing in inputs / editor
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return;
      if (target?.isContentEditable) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const map: Record<string, TabId> = {
        "1": "setup",
        "2": "optimize",
        "3": "explorer",
        "4": "shaping",
        "5": "schematic",
        "6": "health",
      };
      const tab = map[e.key];
      if (!tab) return;
      if (
        !isApplied &&
        tab !== "setup" &&
        tab !== "schematic" &&
        tab !== "explorer" &&
        tab !== "health"
      )
        return;
      setActiveTab(tab);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [isApplied]);

  return (
    <main className="flex h-screen flex-col bg-bg text-fg">
      <Topbar activeTab={activeTab} onTabChange={setActiveTab} />
      <div className="flex min-h-0 flex-1 overflow-hidden">
        <LeftRail activeTab={activeTab} onTabChange={setActiveTab} />
        <section
          id={`${activeTab}-panel`}
          role="tabpanel"
          aria-labelledby={`${activeTab}-tab`}
          className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden"
        >
          {activeTab === "setup" && <SetupTab appConfig={appConfig} />}
          {activeTab === "schematic" && <SchematicTab />}
          {activeTab === "shaping" && <ScoreShapingTab />}
          {activeTab === "optimize" && <OptimizeTab appConfig={appConfig} />}
          {activeTab === "explorer" && <ExplorerTab />}
          {activeTab === "health" && <HealthTab />}
        </section>
      </div>
    </main>
  );
}
