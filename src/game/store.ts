import { create } from "zustand";
import type { TermReason } from "./guidance";
import { energyForMission, type Mission } from "./scenario";
import { defaultBrain, loadBrain, saveBrain, type Brain, type GymSnap } from "./trainer";

export type CamMode = "chase" | "pad" | "orbit";
export type AppPilot = "autopilot" | "manual" | "train";

export type HudSnap = {
  t: number;
  alt: number;
  engineAlt: number;
  speed: number;
  range: number;
  tiltDeg: number;
  fuel: number;
  phase: string;
  term: TermReason;
  qkpa: number;
  mach: number;
  throttle: number;
  nEngines: number;
  g: number;
  intact: boolean;
  energy: number;
  watch?: boolean;
  watchGen?: number;
  watchIdx?: number;
};

type State = {
  started: boolean;
  pilot: AppPilot;
  mission: Mission;
  cam: CamMode;
  warp: number;
  paused: boolean;
  brain: Brain;
  snap: HudSnap | null;
  gym: GymSnap | null;
  genNote: string;
  seed: number;
  start: (pilot?: AppPilot) => void;
  setPilot: (p: AppPilot) => void;
  setMission: (m: Mission) => void;
  setCam: (c: CamMode) => void;
  cycleWarp: () => void;
  setPaused: (v: boolean) => void;
  setSnap: (s: HudSnap) => void;
  setGym: (g: GymSnap) => void;
  setBrain: (b: Brain) => void;
  setGenNote: (n: string) => void;
  resetBrain: () => void;
  bumpSeed: () => void;
};

function initialBrain(): Brain {
  if (typeof window === "undefined") return defaultBrain();
  try {
    return loadBrain();
  } catch {
    return defaultBrain();
  }
}

export const useSpacey = create<State>((set, get) => ({
  started: false,
  pilot: "train",
  mission: "slam",
  cam: "chase",
  warp: 4,
  paused: false,
  brain: initialBrain(),
  snap: null,
  gym: null,
  genNote: "",
  seed: 42,
  start: (pilot) => {
    const p = pilot ?? get().pilot;
    if (p === "train") {
      set({ started: true, paused: false, pilot: p, mission: "slam", warp: Math.max(4, get().warp) });
    } else if (p === "autopilot") {
      set({ started: true, paused: false, pilot: p, mission: "pad", warp: 1 });
    } else {
      set({ started: true, paused: false, pilot: p });
    }
  },
  setPilot: (pilot) => set({ pilot, started: true, paused: false }),
  setMission: (mission) => {
    const next: Partial<State> = { mission, started: true, paused: false, seed: get().seed + 1 };
    if (get().pilot === "train") {
      const brain = { ...get().brain, energy: energyForMission(mission) };
      saveBrain(brain);
      next.brain = brain;
    }
    set(next);
  },
  setCam: (cam) => set({ cam }),
  cycleWarp: () => set({ warp: get().warp >= 16 ? 1 : get().warp * 4 }),
  setPaused: (paused) => set({ paused }),
  setSnap: (snap) => set({ snap }),
  setGym: (gym) => set({ gym }),
  setBrain: (brain) => {
    saveBrain(brain);
    set({ brain });
  },
  setGenNote: (genNote) => set({ genNote }),
  resetBrain: () => {
    const brain = defaultBrain();
    saveBrain(brain);
    set({ brain, gym: null, genNote: "Net reset to zeros. GNC only.", seed: get().seed + 1 });
  },
  bumpSeed: () => set({ seed: get().seed + 1, started: true, paused: false }),
}));
