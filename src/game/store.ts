import { create } from "zustand";
import type { TermReason } from "./guidance";
import { missionFromEnergy, type Mission } from "./scenario";
import { defaultBrain, loadBrain, saveBrain, type Brain, type GymSnap } from "./trainer";

export const WARP_STEPS = [1, 2, 4, 8, 16] as const;
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
  brainEpoch: number;
  snap: HudSnap | null;
  gym: GymSnap | null;
  genNote: string;
  seed: number;
  start: (pilot?: AppPilot, mission?: Mission) => void;
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

const bootBrain = initialBrain();

export const useSpacey = create<State>((set, get) => ({
  started: false,
  pilot: "train",
  mission: missionFromEnergy(bootBrain.energy),
  cam: "chase",
  warp: 1,
  paused: false,
  brain: bootBrain,
  brainEpoch: 0,
  snap: null,
  gym: null,
  genNote: "",
  seed: 42,
  start: (pilot, mission) => {
    const p = pilot ?? get().pilot;
    if (p === "train") {
      set({
        started: true,
        paused: false,
        pilot: p,
        mission: missionFromEnergy(get().brain.energy),
      });
    } else if (p === "autopilot") {
      const m = mission ?? "pad";
      set({
        started: true,
        paused: false,
        pilot: p,
        mission: m,
        warp: 1,
        cam: m === "slam" ? "pad" : get().cam,
      });
    } else {
      set({ started: true, paused: false, pilot: p });
    }
  },
  setPilot: (pilot) => set({ pilot, started: true, paused: false }),
  setMission: (mission) => {
    if (get().pilot === "train") return;
    set({ mission, started: true, paused: false, seed: get().seed + 1 });
  },
  setCam: (cam) => set({ cam }),
  cycleWarp: () => {
    const w = get().warp;
    const i = WARP_STEPS.indexOf(w as (typeof WARP_STEPS)[number]);
    const next = WARP_STEPS[i >= 0 && i < WARP_STEPS.length - 1 ? i + 1 : 0];
    set({ warp: next });
  },
  setPaused: (paused) => set({ paused }),
  setSnap: (snap) => set({ snap }),
  setGym: (gym) => set({ gym }),
  setBrain: (brain) => {
    saveBrain(brain);
    const next: Partial<State> = { brain };
    if (get().pilot === "train") next.mission = missionFromEnergy(brain.energy);
    set(next);
  },
  setGenNote: (genNote) => set({ genNote }),
  resetBrain: () => {
    const brain = defaultBrain();
    saveBrain(brain);
    set({
      brain,
      gym: null,
      genNote: "Net reset to zeros. 21→8→10 · sep-CMA · gate starts at Pad.",
      seed: get().seed + 1,
      brainEpoch: get().brainEpoch + 1,
      mission: "pad",
    });
  },
  bumpSeed: () => set({ seed: get().seed + 1, started: true, paused: false }),
}));
