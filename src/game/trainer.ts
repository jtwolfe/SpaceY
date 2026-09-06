import { clamp, rng } from "./math";
import { SAVE_KEY, SAVE_VERSION, lightBudget, missionFromEnergy, nextEnergy, snapEnergy, type Mission } from "./scenario";
import { Sim } from "./sim";
import {
  N_HIDDEN_LAYERS_MAX,
  N_HIDDEN_LAYERS_MIN,
  growVec,
  growWeights,
  layersFromLen,
  nWeights,
  topologyOf,
  zeroWeights,
} from "./policy";
import { refTrackCost } from "./reftraj";

export const POP = 64;
/** Hansen default μ = ⌊λ/2⌋. Recombination is log-weighted over this many parents. */
export const MU = Math.floor(POP / 2);
export const ELITE = MU;
export const TRAIL_MAX = 280;
export const GATE_RATE = 0.4;
export const GATE_NEED = 2;
export const EXTRA_LIGHT_TAX = 400;
export const EARLY_LIGHT_TAX = 0.7;
export const EARLY_LIGHT_SLACK_M = 80;
/** Quadratic pad-offset tax on a legal land. Q=12 at 35 m ate the 12k jackpot (14.7k) and a rim land lost to a miss. */
export const LAND_RANGE_QUAD = 5;
export const WOBBLE_TAX = 28;
export const WOBBLE_LAND_TAX = 10;
/** Metres of pad range killed unlit above COAST_ALT_M. */
export const COAST_KILL = 22;
export const COAST_REMAIN = 12;
export const COAST_VH = 6;
export const COAST_LINGER = 0.025;
/** Metres of ballistic miss killed unlit. */
export const COAST_PRED = 16;
/** Seconds of Merlin on during landing. */
export const ENGINE_ON_TAX = 12;
/** padDot-seconds of engines-away (nose at pad) while still short of the disk. */
export const THRUST_AWAY_TAX = 90;
/** Seconds of 3-wide during landing. A brief necessary pulse is cheap; a cruise is not. */
export const THREE_ON_TAX = 35;
/** Extra per second of 3-wide sitting at Merlin min — cluster floor is 1.2× a full single. */
export const THREE_MIN_TAX = 160;
/** tiltDeg / alt and |vh| / alt integrated below COAST_ALT_M — even on a legal land. */
export const LATE_TILT = 420;
export const LATE_VH = 180;
export const SIGMA_MIN = 0.05;
export const SIGMA_MAX = 0.28;
export const GROW_SIGMA = 0.16;
export const GROW_NEED = 3;
export const GROW_MIN_GENS = 5;
export const GROW_PLATEAU_RATE = 0.25;
/** RTLS basin is sparse; allow the 5→6 grow without waiting for pad-like 25%. */
export const GROW_PLATEAU_RATE_RTLS = 0.06;
/** Closest-approach (m) that still counts as a useful RTLS parent. Far misses do not. */
export const RTLS_NEAR_M = 200;
/** Once this many RTLS landers exist, drop misses so a lucky 20 km coast cannot yank the mean. */
export const RTLS_LANDERS_ONLY = 8;

export type Brain = {
  version: number;
  weights: number[];
  sigma: number;
  ps: number[];
  pc: number[];
  diagC: number[];
  energy: number;
  gen: number;
  bestFit: number;
  landRate: number;
  episodes: number;
  lands: number;
};

export function layersForEnergy(energy: number) {
  const e = snapEnergy(energy);
  if (e >= 0.999) return Math.min(5, N_HIDDEN_LAYERS_MAX);
  if (e >= 0.68) return Math.min(3, N_HIDDEN_LAYERS_MAX);
  if (e >= 0.22) return Math.min(2, N_HIDDEN_LAYERS_MAX);
  return N_HIDDEN_LAYERS_MIN;
}

function fitBrainLayers(b: Brain) {
  const want = layersForEnergy(b.energy);
  let L = layersFromLen(b.weights.length);
  if (!L) return;
  // Glide used to cap at 4. A 4-layer RTLS brain is glide-tuned residual on an
  // 80 km entry — damp toward GNC once while growing to the RTLS floor (5).
  const dampGlideOnRtls = b.energy >= 0.85 && L < 5;
  while (L < want && L < N_HIDDEN_LAYERS_MAX) {
    const from = b.weights.length;
    b.weights = growWeights(b.weights);
    b.ps = growVec(b.ps, 0, from);
    b.pc = growVec(b.pc, 0, from);
    b.diagC = growVec(b.diagC, 1, from);
    L = layersFromLen(b.weights.length) || L + 1;
  }
  if (dampGlideOnRtls) {
    for (let i = 0; i < b.weights.length; i++) b.weights[i] *= 0.08;
    b.ps.fill(0);
    b.pc.fill(0);
    b.diagC.fill(1);
    b.sigma = clamp(Math.max(b.sigma, 0.12) * 1.1, SIGMA_MIN, 0.22);
  }
}

function zeros(n: number) {
  return new Array(n).fill(0);
}
function ones(n: number) {
  return new Array(n).fill(1);
}

function fitLen(v: number[] | undefined, n: number, fill: number) {
  if (!Array.isArray(v) || v.length !== n) return new Array(n).fill(fill);
  return v.map(Number);
}

export function defaultBrain(): Brain {
  const n = nWeights(N_HIDDEN_LAYERS_MIN);
  return {
    version: SAVE_VERSION,
    weights: zeroWeights(N_HIDDEN_LAYERS_MIN),
    sigma: 0.2,
    ps: zeros(n),
    pc: zeros(n),
    diagC: ones(n),
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
    const L = Array.isArray(parsed.weights) ? layersFromLen(parsed.weights.length) : 0;
    if (parsed.version !== SAVE_VERSION || !L) return defaultBrain();
    const n = parsed.weights!.length;
    const brain: Brain = {
      ...defaultBrain(),
      ...parsed,
      weights: parsed.weights!.map(Number),
      sigma: clamp(Number(parsed.sigma) || 0.2, SIGMA_MIN, SIGMA_MAX),
      ps: fitLen(parsed.ps, n, 0),
      pc: fitLen(parsed.pc, n, 0),
      diagC: fitLen(parsed.diagC, n, 1),
      energy: snapEnergy(Number(parsed.energy) || 0),
    };
    fitBrainLayers(brain);
    return brain;
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

function logWeights(mu: number) {
  const raw = Array.from({ length: mu }, (_, i) => Math.log(mu + 0.5) - Math.log(i + 1));
  const sum = raw.reduce((a, b) => a + b, 0);
  const w = raw.map((x) => x / sum);
  const muEff = 1 / w.reduce((a, b) => a + b * b, 0);
  return { w, muEff };
}

function cmaConsts(n: number, muEff: number) {
  const cs = (muEff + 2) / (n + muEff + 5);
  const damps = 1 + cs + 2 * Math.max(0, Math.sqrt((muEff - 1) / (n + 1)) - 1);
  const cc = (4 + muEff / n) / (n + 4 + (2 * muEff) / n);
  let c1 = 2 / ((n + 1.3) ** 2 + muEff);
  let cmu = Math.min(1 - c1, (2 * (muEff - 2 + 1 / muEff)) / ((n + 2) ** 2 + muEff));
  if (!(cmu > 0)) cmu = 0;
  const sep = (n + 2) / 3;
  c1 *= sep;
  cmu *= sep;
  const s = c1 + cmu;
  if (s > 0.99) {
    c1 *= 0.99 / s;
    cmu *= 0.99 / s;
  }
  const chiN = Math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n));
  return { cs, damps, cc, c1, cmu, chiN };
}

export function sampleWeights(mean: number[], sigma: number, rand: () => number, diagC?: number[]): { x: number[]; z: number[] } {
  const n = mean.length;
  const z = Array.from({ length: n }, () => boxMuller(rand));
  const x = new Array(n);
  for (let j = 0; j < n; j++) {
    const d = Math.sqrt(Math.max(1e-12, diagC?.[j] ?? 1));
    x[j] = clamp(mean[j] + sigma * d * z[j], -2.8, 2.8);
  }
  return { x, z };
}

export type Agent = {
  weights: number[];
  z: number[];
  sim: Sim;
  fit: number;
  term: string;
  landed: boolean;
  trail: Float32Array;
  trailLen: number;
  trailCursor: number;
};

function earlyLightM(sim: Sim): number {
  if (sim.landingLightAlt <= 0) return 0;
  return Math.max(0, sim.landingLightAlt - sim.landingSLight - EARLY_LIGHT_SLACK_M);
}

function clusterTax(sim: Sim): number {
  return (
    ENGINE_ON_TAX * sim.engineOnT +
    THRUST_AWAY_TAX * sim.thrustAwayShortT +
    THREE_ON_TAX * sim.threeOnT +
    THREE_MIN_TAX * sim.threeMinT
  );
}

function poseTax(sim: Sim): number {
  const n = sim.nav;
  const range = n.rangeH;
  const tiltDeg = (n.tilt * 180) / Math.PI;
  const near = range < 28 ? 12 * tiltDeg : 3 * tiltDeg;
  return near + clusterTax(sim);
}

/** Reward range and ballistic miss that died on the unlit high-q coast; tax leftover range × vh at light. */
export function coastScore(sim: Sim): number {
  const remain = sim.coastLatched ? sim.coastRemain : sim.nav.rangeH;
  const vh = sim.coastLatched ? sim.coastVh : Math.hypot(sim.v.x, sim.v.y);
  const kill = sim.coastLatched ? sim.coastKill : Math.max(0, sim.spawnRange - remain);
  const predKill = sim.coastLatched ? sim.coastPredKill : Math.max(0, sim.spawnPredMiss - sim.coastPredRemain);
  // Pad / 2 km spawn inside 120 m: scale = 1 (unchanged). Glide ~11 km would otherwise
  // let linger (range × vh × t) eat the 12k land jackpot so a destroy outranks a land.
  const scale = 120 / Math.max(120, sim.spawnRange || 120);
  const linger = Math.min(COAST_LINGER * sim.coastLingerT * scale, 2_400);
  return (
    COAST_KILL * kill * scale +
    COAST_PRED * predKill * scale -
    COAST_REMAIN * remain * scale -
    COAST_VH * vh -
    linger
  );
}

/** Late TVC hook: tilt and horizontal speed weighted by 1/alt in the last 400 m. */
export function lateLeanTax(sim: Sim): number {
  return LATE_TILT * sim.lateTiltT + LATE_VH * sim.lateVhT;
}

/** Honest score: a land always beats a miss. Misses that recede after closest approach lose to tight slams. */
export function scoreSim(sim: Sim): number {
  const n = sim.nav;
  const range = n.rangeH;
  const recede = Math.max(0, range - sim.minRange);
  const climb = sim.climbT;
  const extra = Math.max(0, sim.engine.lights - lightBudget(sim.energy));
  const lightTax = extra * EXTRA_LIGHT_TAX + earlyLightM(sim) * EARLY_LIGHT_TAX;
  const pose = poseTax(sim);
  const track = refTrackCost(sim.ref, sim.p, sim.v);
  const wobble = sim.wobbleT;
  const coast = coastScore(sim);
  const late = lateLeanTax(sim);
  if (sim.term === "landed") {
    return (
      12_000 +
      n.fuel * 0.08 -
      sim.t * 2 -
      n.speed * 40 -
      LAND_RANGE_QUAD * range * range -
      lightTax -
      clusterTax(sim) -
      WOBBLE_LAND_TAX * wobble +
      coast -
      late
    );
  }
  const speed = n.speed;
  if (sim.term === "miss") {
    let miss =
      -8 * range - 6 * recede - 10 * speed - 0.4 * Math.max(0, n.engineAlt) - 40 * climb - lightTax - pose - track - WOBBLE_TAX * wobble + coast - late;
    if (sim.energy >= 0.85) {
      miss += -8 * recede + 10 * Math.max(0, RTLS_NEAR_M - sim.minRange);
    }
    return miss;
  }
  if (sim.term === "destroyed") {
    return -7_000 - range * 0.4 - speed * 4 - 3 * recede - 15 * climb - lightTax - pose - track - 0.4 * WOBBLE_TAX * wobble + 0.5 * coast - 0.5 * late;
  }
  return -5_000 - range * 0.3 - speed * 2 - n.engineAlt * 0.05 - 4 * recede - 20 * climb - lightTax - pose - track - 0.5 * WOBBLE_TAX * wobble + 0.5 * coast - 0.5 * late;
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
  nLayers: number;
  nLayersMax: number;
  topology: string;
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
  growStreak: number;
  growNeed: number;
  mu: number;
  lights: number;
  lightBudget: number;
  refDist: number;
  refOk: boolean;
  coastKill: number;
  coastRemain: number;
  coastLatched: boolean;
  coastPredKill: number;
  engineOnT: number;
  thrustAwayShortT: number;
  threeOnT: number;
  threeMinT: number;
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

/** Pick CMA parents. Pad / 2 km / Glide: landers first, fill with next-best.
 *  RTLS: a high-fit 20 km coast-kill must not sit in μ next to one lander. */
export function selectParents(agents: Agent[], energy: number): Agent[] {
  const ranked = [...agents].sort((a, b) => b.fit - a.fit);
  const landers = ranked.filter((a) => a.landed);
  const rest = ranked.filter((a) => !a.landed);
  if (energy >= 0.85) {
    if (landers.length >= RTLS_LANDERS_ONLY) return landers.slice(0, MU);
    const near = rest.filter((a) => a.sim.minRange < RTLS_NEAR_M);
    const pool = [...landers, ...near];
    if (pool.length > 0) return pool.slice(0, Math.min(MU, pool.length));
    const closest = [...rest].sort((a, b) => a.sim.minRange - b.sim.minRange);
    return closest.slice(0, Math.min(8, closest.length));
  }
  const pool = landers.length >= MU ? landers : [...landers, ...rest];
  return pool.slice(0, Math.min(MU, pool.length));
}

export class Trainer {
  brain: Brain;
  agents: Agent[] = [];
  seed0 = 1;
  genLands = 0;
  streak = 0;
  growStreak = 0;
  gensOnTopo = 0;
  heroIdx = 0;
  doneCount = 0;
  watchCursor = 0;
  watchIdx = 0;
  watchGen = 0;
  unlocked: Mission | null = null;
  grown: string | null = null;

  constructor(brain?: Brain) {
    this.brain = brain ?? defaultBrain();
    this.brain.energy = snapEnergy(this.brain.energy);
    this.syncCmaLen();
  }

  private syncCmaLen() {
    if (!layersFromLen(this.brain.weights.length)) this.brain.weights = zeroWeights(N_HIDDEN_LAYERS_MIN);
    let n = this.brain.weights.length;
    this.brain.ps = fitLen(this.brain.ps, n, 0);
    this.brain.pc = fitLen(this.brain.pc, n, 0);
    this.brain.diagC = fitLen(this.brain.diagC, n, 1);
    this.brain.sigma = clamp(this.brain.sigma, SIGMA_MIN, SIGMA_MAX);
    fitBrainLayers(this.brain);
    n = this.brain.weights.length;
    this.brain.ps = fitLen(this.brain.ps, n, 0);
    this.brain.pc = fitLen(this.brain.pc, n, 0);
    this.brain.diagC = fitLen(this.brain.diagC, n, 1);
  }

  beginGen() {
    const rand = rng(this.seed0 + this.brain.gen * 997);
    this.agents = [];
    this.genLands = 0;
    this.doneCount = 0;
    this.heroIdx = 0;
    const energy = snapEnergy(this.brain.energy);
    this.brain.energy = energy;
    this.syncCmaLen();
    const destroy = energy > 0.3;
    const mean = this.brain.weights;
    const n = mean.length;
    for (let i = 0; i < POP; i++) {
      const sampled = i === 0 ? { x: [...mean], z: zeros(n) } : sampleWeights(mean, this.brain.sigma, rand, this.brain.diagC);
      const seed = this.seed0 + this.brain.gen * 1009 + i * 17 + 3;
      const sim = Sim.start(energy, seed, { destroy, pilot: "student", weights: sampled.x });
      this.agents.push({
        weights: sampled.x,
        z: sampled.z,
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
    const L = layersFromLen(brain.weights?.length ?? 0);
    this.brain = {
      ...defaultBrain(),
      ...brain,
      weights: L ? [...brain.weights] : zeroWeights(N_HIDDEN_LAYERS_MIN),
      energy: snapEnergy(brain.energy ?? 0),
      sigma: clamp(brain.sigma || 0.2, SIGMA_MIN, SIGMA_MAX),
    };
    this.syncCmaLen();
    this.streak = 0;
    this.growStreak = 0;
    this.gensOnTopo = 0;
    this.watchCursor = 0;
    this.unlocked = null;
    this.grown = null;
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
    const recede = Math.max(0, n.rangeH - a.sim.minRange);
    const extra = Math.max(0, a.sim.engine.lights - lightBudget(a.sim.energy));
    let s =
      -8 * n.rangeH -
      6 * recede -
      10 * n.speed -
      0.35 * Math.max(0, n.engineAlt) -
      40 * a.sim.climbT -
      extra * EXTRA_LIGHT_TAX -
      earlyLightM(a.sim) * EARLY_LIGHT_TAX -
      poseTax(a.sim) -
      refTrackCost(a.sim.ref, a.sim.p, a.sim.v) -
      WOBBLE_TAX * a.sim.wobbleT +
      coastScore(a.sim) -
      lateLeanTax(a.sim);
    if (a.sim.energy >= 0.85) {
      s += -8 * recede + 10 * Math.max(0, RTLS_NEAR_M - a.sim.minRange);
    }
    return s;
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
    const nW = this.brain.weights.length;
    const nL = layersFromLen(nW) || N_HIDDEN_LAYERS_MIN;
    return {
      gen: this.brain.gen,
      energy: this.brain.energy,
      sigma: this.brain.sigma,
      landRate: this.brain.landRate,
      bestFit: this.brain.bestFit,
      episodes: this.brain.episodes,
      idxLive: this.watchIdx,
      nWeights: nW,
      nLayers: nL,
      nLayersMax: N_HIDDEN_LAYERS_MAX,
      topology: topologyOf(nL),
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
      growStreak: this.growStreak,
      growNeed: GROW_NEED,
      mu: MU,
      lights: vis.engine.lights,
      lightBudget: lightBudget(vis.energy),
      refDist: vis.lastRefDist,
      refOk: !!vis.goal,
      coastKill: vis.coastKill,
      coastRemain: vis.coastLatched ? vis.coastRemain : vis.nav.rangeH,
      coastLatched: vis.coastLatched,
      coastPredKill: vis.coastPredKill,
      engineOnT: vis.engineOnT,
      thrustAwayShortT: vis.thrustAwayShortT,
      threeOnT: vis.threeOnT,
      threeMinT: vis.threeMinT,
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

  /** Hansen (μ/μ_w, λ)-CMA-ES with diagonal C (sep-CMA) and CSA on σ. */
  private cmaUpdate(parents: Agent[]) {
    const n = this.brain.weights.length;
    const mu = Math.max(1, Math.min(MU, parents.length));
    const { w, muEff } = logWeights(mu);
    const { cs, damps, cc, c1, cmu, chiN } = cmaConsts(n, muEff);
    const m = this.brain.weights;
    const sigma = this.brain.sigma;
    const diagC = this.brain.diagC;
    const d = new Array(n);
    for (let j = 0; j < n; j++) d[j] = Math.sqrt(Math.max(1e-12, diagC[j]));

    const zW = zeros(n);
    for (let i = 0; i < mu; i++) {
      const z = parents[i].z.length === n ? parents[i].z : zeros(n);
      for (let j = 0; j < n; j++) zW[j] += w[i] * z[j];
    }

    const mean = new Array(n);
    for (let j = 0; j < n; j++) mean[j] = clamp(m[j] + sigma * d[j] * zW[j], -2.8, 2.8);

    const ps = this.brain.ps;
    const pc = this.brain.pc;
    let ps2 = 0;
    const csMix = Math.sqrt(cs * (2 - cs) * muEff);
    for (let j = 0; j < n; j++) {
      ps[j] = (1 - cs) * ps[j] + csMix * zW[j];
      ps2 += ps[j] * ps[j];
    }
    const gen = this.brain.gen + 1;
    const denom = Math.sqrt(Math.max(1e-12, 1 - (1 - cs) ** (2 * gen))) * chiN;
    const hsig = Math.sqrt(ps2) / denom < 1.4 + 2 / (n + 1) ? 1 : 0;
    const nextSigma = clamp(sigma * Math.exp((cs / damps) * (Math.sqrt(ps2) / chiN - 1)), SIGMA_MIN, SIGMA_MAX);

    const ccMix = hsig * Math.sqrt(cc * (2 - cc) * muEff);
    for (let j = 0; j < n; j++) {
      pc[j] = (1 - cc) * pc[j] + ccMix * d[j] * zW[j];
      let rankMu = 0;
      for (let i = 0; i < mu; i++) {
        const z = parents[i].z.length === n ? parents[i].z : zeros(n);
        const y = d[j] * z[j];
        rankMu += w[i] * y * y;
      }
      const stall = (1 - hsig) * cc * (2 - cc) * diagC[j];
      diagC[j] = clamp((1 - c1 - cmu) * diagC[j] + c1 * (pc[j] * pc[j] + stall) + cmu * rankMu, 1e-8, 25);
    }

    this.brain.weights = mean;
    this.brain.sigma = nextSigma;
    this.brain.ps = ps;
    this.brain.pc = pc;
    this.brain.diagC = diagC;
  }

  /** Insert a zero residual block and expand CMA state (mean, C, paths) with it. Mean forward is unchanged. */
  growLayer() {
    const L = layersFromLen(this.brain.weights.length);
    if (!L || L >= N_HIDDEN_LAYERS_MAX) return false;
    const from = this.brain.weights.length;
    this.brain.weights = growWeights(this.brain.weights);
    this.brain.ps = growVec(this.brain.ps, 0, from);
    this.brain.pc = growVec(this.brain.pc, 0, from);
    this.brain.diagC = growVec(this.brain.diagC, 1, from);
    this.syncCmaLen();
    this.growStreak = 0;
    this.gensOnTopo = 0;
    const next = layersFromLen(this.brain.weights.length);
    this.grown = `${topologyOf(next)} · ${this.brain.weights.length} w`;
    return true;
  }

  private tryGrowPlateau() {
    const L = layersFromLen(this.brain.weights.length);
    if (!L || L >= N_HIDDEN_LAYERS_MAX) {
      this.growStreak = 0;
      return;
    }
    if (this.brain.sigma > GROW_SIGMA || this.gensOnTopo < GROW_MIN_GENS) {
      this.growStreak = 0;
      return;
    }
    if (this.brain.landRate < (this.brain.energy >= 0.85 ? GROW_PLATEAU_RATE_RTLS : GROW_PLATEAU_RATE)) {
      this.growStreak = 0;
      return;
    }
    this.growStreak += 1;
    if (this.growStreak < GROW_NEED) return;
    if (!this.growLayer()) return;
    this.brain.sigma = clamp(Math.max(this.brain.sigma, 0.08) * 1.12, SIGMA_MIN, 0.2);
  }

  finishGen() {
    this.unlocked = null;
    this.grown = null;
    const parents = selectParents(this.agents, this.brain.energy);
    if (parents.length) this.cmaUpdate(parents);

    this.brain.gen += 1;
    this.gensOnTopo += 1;
    this.brain.bestFit = Math.max(this.brain.bestFit, parents[0]?.fit ?? -1e9);
    const rate = this.genLands / POP;
    this.brain.landRate = this.brain.landRate * 0.55 + rate * 0.45;

    let unlocked = false;
    if (rate >= GATE_RATE) {
      this.streak += 1;
      const nxt = nextEnergy(this.brain.energy);
      if (this.streak >= GATE_NEED && nxt != null) {
        this.brain.energy = nxt;
        this.streak = 0;
        this.brain.landRate = 0;
        this.brain.sigma = clamp(Math.max(this.brain.sigma, 0.12) * 1.1, SIGMA_MIN, 0.22);
        this.unlocked = missionFromEnergy(nxt);
        unlocked = true;
      }
    } else {
      this.streak = 0;
    }
    if (unlocked) {
      this.growLayer();
      // 2 km residual lofts a glide suicide. Start glide near GNC. RTLS damp
      // lives in fitBrainLayers so a 4/4 already-unlocked brain is damped too.
      if (this.unlocked === "glide") {
        for (let i = 0; i < this.brain.weights.length; i++) this.brain.weights[i] *= 0.08;
        this.brain.ps.fill(0);
        this.brain.pc.fill(0);
        this.brain.diagC.fill(1);
      }
    } else this.tryGrowPlateau();
    saveBrain(this.brain);
    return parents[0];
  }
}

