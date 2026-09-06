import { clamp, rng } from "./math";
import { SAVE_KEY, SAVE_VERSION, missionFromEnergy, nextEnergy, snapEnergy, type Mission } from "./scenario";
import { Sim } from "./sim";
import { N_WEIGHTS, zeroWeights } from "./policy";

export const POP = 12;
export const ELITE = 3;
export const TRAIL_MAX = 280;
export const GATE_RATE = 0.4;
export const GATE_NEED = 2;
/** Points per metre of pad offset on a land. Was 8 — too flat vs the +12000 jackpot. */
export const LAND_RANGE_TAX = 80;

export type Brain = {
  version: number;
  weights: number[];
  sigma: number;
  energy: number;
  gen: number;
  bestFit: number;
  landRate: number;
  episodes: number;
  lands: number;
};

export function defaultBrain(): Brain {
  return {
    version: SAVE_VERSION,
    weights: zeroWeights(),
    sigma: 0.2,
    energy: 0,
    gen: 0,
    bestFit: -1e9,
    landRate: 0,
    episodes: 0,
    lands: 0,
  };
}

export function loadBrain(): Brain {
  try {
    const raw = localStorage.getItem(SAVE_KEY);
    if (!raw) return defaultBrain();
    const parsed = JSON.parse(raw) as Partial<Brain> & { mean?: number[] };
    if (parsed.version !== SAVE_VERSION || !Array.isArray(parsed.weights) || parsed.weights.length !== N_WEIGHTS) {
      return defaultBrain();
    }
    return {
      ...defaultBrain(),
      ...parsed,
      weights: parsed.weights.map(Number),
      sigma: Math.max(0.12, Number(parsed.sigma) || 0.2),
      energy: snapEnergy(Number(parsed.energy) || 0),
    };
  } catch {
    return defaultBrain();
  }
}

export function saveBrain(b: Brain) {
  try {
    localStorage.setItem(SAVE_KEY, JSON.stringify(b));
  } catch {
    /* private mode */
  }
}

function boxMuller(rand: () => number) {
  const u = Math.max(1e-9, rand());
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function sampleWeights(mean: number[], sigma: number, rand: () => number): number[] {
  return mean.map((m) => clamp(m + boxMuller(rand) * sigma, -2.8, 2.8));
}

export type Agent = {
  weights: number[];
  sim: Sim;
  fit: number;
  term: string;
  landed: boolean;
  trail: Float32Array;
  trailLen: number;
  trailCursor: number;
};

/** Honest score: a land always beats a miss. Closer lands beat rim lands. */
export function scoreSim(sim: Sim): number {
  const n = sim.nav;
  if (sim.term === "landed") {
    return 12_000 + n.fuel * 0.08 - sim.t * 2 - n.speed * 40 - n.rangeH * LAND_RANGE_TAX;
  }
  const range = n.rangeH;
  const speed = n.speed;
  const tilt = (n.tilt * 180) / Math.PI;
  if (sim.term === "miss") {
    return -8 * range - 14 * speed - 18 * tilt - 0.4 * Math.max(0, n.engineAlt);
  }
  if (sim.term === "destroyed") {
    return -7_000 - range * 0.4 - speed * 4;
  }
  return -5_000 - range * 0.3 - speed * 2 - n.engineAlt * 0.05;
}

export type GymSnap = {
  gen: number;
  energy: number;
  sigma: number;
  landRate: number;
  bestFit: number;
  episodes: number;
  idxLive: number;
  nWeights: number;
  nLive: number;
  nLand: number;
  nDead: number;
  meanAbsY: number;
  weightNorm: number;
  simT: number;
  watchIdx: number;
  watchGen: number;
  hidden: number[];
  outputs: number[];
  stage: Mission;
  gateStreak: number;
  gateNeed: number;
  gateRate: number;
  pop: {
    fit: number;
    term: string;
    landed: boolean;
    alive: boolean;
    range: number;
    alt: number;
    hero: boolean;
  }[];
};

export class Trainer {
  brain: Brain;
  agents: Agent[] = [];
  seed0 = 1;
  genLands = 0;
  streak = 0;
  heroIdx = 0;
  doneCount = 0;
  watchCursor = 0;
  watchIdx = 0;
  watchGen = 0;
  unlocked: Mission | null = null;

  constructor(brain?: Brain) {
    this.brain = brain ?? defaultBrain();
    this.brain.energy = snapEnergy(this.brain.energy);
  }

  beginGen() {
    const rand = rng(this.seed0 + this.brain.gen * 997);
    this.agents = [];
    this.genLands = 0;
    this.doneCount = 0;
    this.heroIdx = 0;
    const energy = snapEnergy(this.brain.energy);
    this.brain.energy = energy;
    const destroy = energy > 0.3;
    for (let i = 0; i < POP; i++) {
      const weights = i === 0 ? [...this.brain.weights] : sampleWeights(this.brain.weights, this.brain.sigma, rand);
      const seed = this.seed0 + this.brain.gen * 1009 + i * 17 + 3;
      const sim = Sim.start(energy, seed, { destroy, pilot: "student", weights });
      this.agents.push({
        weights,
        sim,
        fit: -1e9,
        term: "none",
        landed: false,
        trail: new Float32Array(TRAIL_MAX * 3),
        trailLen: 0,
        trailCursor: 0,
      });
    }
  }

  applyBrain(brain: Brain) {
    this.brain = {
      ...defaultBrain(),
      ...brain,
      weights: brain.weights?.length === N_WEIGHTS ? [...brain.weights] : zeroWeights(),
      energy: snapEnergy(brain.energy ?? 0),
      sigma: Math.max(0.12, brain.sigma || 0.2),
    };
    this.streak = 0;
    this.watchCursor = 0;
    this.unlocked = null;
    this.beginGen();
  }

  stepAll(dt: number) {
    for (let i = 0; i < this.agents.length; i++) {
      const a = this.agents[i];
      if (a.sim.terminated()) continue;
      a.sim.step(dt);
      if ((a.trailCursor & 1) === 0) this.pushTrail(a);
      a.trailCursor += 1;
      if (a.sim.terminated()) {
        a.fit = scoreSim(a.sim);
        a.term = a.sim.term;
        a.landed = a.sim.term === "landed";
        this.doneCount += 1;
        this.brain.episodes += 1;
        if (a.landed) {
          this.genLands += 1;
          this.brain.lands += 1;
        }
      }
    }
    this.pickHero();
  }

  allDone() {
    return this.doneCount >= this.agents.length && this.agents.length > 0;
  }

  private pushTrail(a: Agent) {
    const p = a.sim.p;
    const i = (a.trailLen % TRAIL_MAX) * 3;
    a.trail[i] = p.x;
    a.trail[i + 1] = p.y;
    a.trail[i + 2] = p.z;
    a.trailLen += 1;
  }

  private pickHero() {
    let best = 0;
    let bestKey = Infinity;
    for (let i = 0; i < this.agents.length; i++) {
      const a = this.agents[i];
      const live = !a.sim.terminated();
      const key = live ? a.sim.nav.rangeH : 1e7 + Math.abs(a.fit);
      if (a.landed) {
        best = i;
        bestKey = -1;
        break;
      }
      if (key < bestKey) {
        bestKey = key;
        best = i;
      }
    }
    this.heroIdx = best;
  }

  /** Fresh 1× viewer: clone a genome from the live generation, not a warp-speed body. */
  takeWatch() {
    if (!this.agents.length) this.beginGen();
    const n = Math.max(1, this.agents.length);
    const idx = ((this.watchCursor % n) + n) % n;
    this.watchCursor += 1;
    this.watchIdx = idx;
    this.watchGen = this.brain.gen;
    const a = this.agents[idx];
    return {
      weights: [...a.weights],
      energy: this.brain.energy,
      gen: this.brain.gen,
      idx,
      seed: this.seed0 + this.brain.gen * 7919 + idx * 17 + this.watchCursor * 13,
    };
  }

  hero(): Sim {
    const a = this.agents[this.heroIdx] ?? this.agents[0];
    if (!a) {
      this.beginGen();
      return this.agents[0].sim;
    }
    return a.sim;
  }

  private liveFit(a: Agent) {
    if (a.sim.terminated()) return a.fit;
    const n = a.sim.nav;
    return -n.rangeH - 14 * n.speed - 18 * ((n.tilt * 180) / Math.PI) - 0.35 * Math.max(0, n.engineAlt);
  }

  snap(view?: Sim): GymSnap {
    if (!this.agents.length) this.beginGen();
    let vis = view ?? this.hero();
    let bestMag = -1;
    let nLive = 0;
    let nLand = 0;
    let nDead = 0;
    let absY = 0;
    let simT = 0;
    for (const a of this.agents) {
      simT = Math.max(simT, a.sim.t);
      const mag = a.sim.lastY.reduce((s, v) => s + Math.abs(v), 0);
      absY += mag;
      if (a.landed) nLand += 1;
      else if (a.sim.terminated()) nDead += 1;
      else nLive += 1;
      if (!view && !a.sim.terminated() && mag > bestMag) {
        bestMag = mag;
        vis = a.sim;
      }
    }
    if (!view && bestMag <= 0) vis = this.agents[1]?.sim ?? vis;
    let wNorm = 0;
    for (const w of this.brain.weights) wNorm += w * w;
    return {
      gen: this.brain.gen,
      energy: this.brain.energy,
      sigma: this.brain.sigma,
      landRate: this.brain.landRate,
      bestFit: this.brain.bestFit,
      episodes: this.brain.episodes,
      idxLive: this.watchIdx,
      nWeights: N_WEIGHTS,
      nLive,
      nLand,
      nDead,
      meanAbsY: this.agents.length ? absY / this.agents.length : 0,
      weightNorm: Math.sqrt(wNorm),
      simT,
      watchIdx: this.watchIdx,
      watchGen: this.watchGen,
      hidden: [...(vis.lastHidden ?? [])],
      outputs: [...(vis.lastY ?? [])],
      stage: missionFromEnergy(this.brain.energy),
      gateStreak: this.streak,
      gateNeed: GATE_NEED,
      gateRate: GATE_RATE,
      pop: this.agents.map((a, i) => ({
        fit: this.liveFit(a),
        term: a.sim.terminated() ? a.term : "fly",
        landed: a.landed,
        alive: !a.sim.terminated(),
        range: a.sim.nav.rangeH,
        alt: a.sim.nav.engineAlt,
        hero: i === this.watchIdx,
      })),
    };
  }

  finishGen() {
    this.unlocked = null;
    const ranked = [...this.agents].sort((a, b) => b.fit - a.fit);
    const landers = ranked.filter((a) => a.landed);
    const elite = (landers.length ? landers : ranked).slice(0, ELITE);
    const mean = Array.from({ length: N_WEIGHTS }, (_, k) => {
      let s = 0;
      for (const e of elite) s += e.weights[k];
      return s / elite.length;
    });
    let varSum = 0;
    for (const e of elite) {
      for (let k = 0; k < N_WEIGHTS; k++) {
        const d = e.weights[k] - mean[k];
        varSum += d * d;
      }
    }
    const sigma = clamp(Math.sqrt(varSum / (elite.length * N_WEIGHTS)) * 1.25 + 0.09, 0.12, 0.48);
    this.brain.weights = mean;
    this.brain.sigma = sigma;
    this.brain.gen += 1;
    this.brain.bestFit = Math.max(this.brain.bestFit, elite[0]?.fit ?? -1e9);
    const rate = this.genLands / POP;
    this.brain.landRate = this.brain.landRate * 0.55 + rate * 0.45;

    if (rate >= GATE_RATE) {
      this.streak += 1;
      const nxt = nextEnergy(this.brain.energy);
      if (this.streak >= GATE_NEED && nxt != null) {
        this.brain.energy = nxt;
        this.streak = 0;
        this.brain.landRate = 0;
        this.brain.sigma = Math.max(this.brain.sigma, 0.16);
        this.unlocked = missionFromEnergy(nxt);
      }
    } else {
      this.streak = 0;
    }
    saveBrain(this.brain);
    return elite[0];
  }
}
