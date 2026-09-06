import {
  GLIDE_FUEL_KG,
  HOVER_FUEL_KG,
  START_FUEL_KG,
  SUICIDE_FUEL_KG,
} from "./constants";
import { lerp, Quat, rng, Vec3 } from "./math";

export type Envelope = {
  energy: number;
  alt: number;
  ve: number;
  vn: number;
  vu: number;
  east: number;
  north: number;
  fuel: number;
  tilt: number;
  tailFirst: boolean;
  wind: number;
  timeout: number;
};

type Knot = Envelope;

const KNOTS: Knot[] = [
  {
    energy: 0,
    alt: 250,
    ve: 0,
    vn: 0,
    vu: -15,
    east: 0,
    north: 0,
    fuel: HOVER_FUEL_KG,
    tilt: 0,
    tailFirst: false,
    wind: 0.15,
    timeout: 45,
  },
  {
    energy: 0.22,
    alt: 2050,
    ve: 28,
    vn: 0,
    vu: -120,
    east: 160,
    north: 0,
    fuel: SUICIDE_FUEL_KG,
    tilt: 0,
    tailFirst: false,
    wind: 0.25,
    timeout: 90,
  },
  {
    energy: 0.42,
    alt: 2100,
    ve: 35,
    vn: 22,
    vu: -130,
    east: 180,
    north: 140,
    fuel: SUICIDE_FUEL_KG,
    tilt: (10 * Math.PI) / 180,
    tailFirst: false,
    wind: 1,
    timeout: 90,
  },
  {
    energy: 0.68,
    alt: 20_500,
    ve: -365,
    vn: 8,
    vu: -62,
    east: 11_000,
    north: 200,
    fuel: GLIDE_FUEL_KG,
    tilt: 0,
    tailFirst: true,
    wind: 1,
    timeout: 200,
  },
  {
    energy: 1,
    alt: 80_000,
    ve: -2025,
    vn: 25,
    vu: -268,
    east: 62_000,
    north: 400,
    fuel: START_FUEL_KG,
    tilt: 0,
    tailFirst: true,
    wind: 1,
    timeout: 420,
  },
];

function lerpKnot(a: Knot, b: Knot, t: number): Envelope {
  return {
    energy: lerp(a.energy, b.energy, t),
    alt: lerp(a.alt, b.alt, t),
    ve: lerp(a.ve, b.ve, t),
    vn: lerp(a.vn, b.vn, t),
    vu: lerp(a.vu, b.vu, t),
    east: lerp(a.east, b.east, t),
    north: lerp(a.north, b.north, t),
    fuel: lerp(a.fuel, b.fuel, t),
    tilt: lerp(a.tilt, b.tilt, t),
    tailFirst: t > 0.5 ? b.tailFirst : a.tailFirst,
    wind: lerp(a.wind, b.wind, t),
    timeout: lerp(a.timeout, b.timeout, t),
  };
}

export function envelopeAt(energy: number): Envelope {
  const e = Math.min(1, Math.max(0, energy));
  for (let i = 0; i < KNOTS.length - 1; i++) {
    const a = KNOTS[i];
    const b = KNOTS[i + 1];
    if (e >= a.energy && e <= b.energy) {
      const t = (e - a.energy) / Math.max(1e-9, b.energy - a.energy);
      return lerpKnot(a, b, t);
    }
  }
  return { ...KNOTS[KNOTS.length - 1] };
}

export type Mission = "pad" | "slam" | "glide" | "rtls";

export const LADDER: { id: Mission; energy: number; label: string }[] = [
  { id: "pad", energy: 0, label: "Pad" },
  { id: "slam", energy: 0.22, label: "2 km" },
  { id: "glide", energy: 0.68, label: "Glide" },
  { id: "rtls", energy: 1, label: "RTLS" },
];

export function energyForMission(m: Mission): number {
  switch (m) {
    case "pad":
      return 0;
    case "slam":
      return 0.22;
    case "glide":
      return 0.68;
    case "rtls":
      return 1;
  }
}

/** Snap continuous energy onto the nearest named training rung. */
export function snapEnergy(energy: number): number {
  const e = Math.min(1, Math.max(0, energy));
  let best = LADDER[0].energy;
  let bestD = Infinity;
  for (const r of LADDER) {
    const d = Math.abs(r.energy - e);
    if (d < bestD) {
      bestD = d;
      best = r.energy;
    }
  }
  return best;
}

export function missionFromEnergy(energy: number): Mission {
  const e = snapEnergy(energy);
  let m: Mission = "pad";
  for (const r of LADDER) if (e + 1e-9 >= r.energy) m = r.id;
  return m;
}

export function nextEnergy(energy: number): number | null {
  const e = snapEnergy(energy);
  for (const r of LADDER) if (r.energy > e + 1e-9) return r.energy;
  return null;
}

export function missionLabel(energy: number) {
  switch (missionFromEnergy(energy)) {
    case "pad":
      return "Pad slam";
    case "slam":
      return "2 km";
    case "glide":
      return "Glide";
    case "rtls":
      return "RTLS";
  }
}

export type Spawn = {
  p: Vec3;
  v: Vec3;
  q: Quat;
  omega: Vec3;
  fuel: number;
  wind: number;
  timeout: number;
  energy: number;
};

export function spawnAt(energy: number, seed: number): Spawn {
  const env = envelopeAt(energy);
  const rand = rng(seed);
  const jitter = (amp: number) => amp * (rand() * 2 - 1);

  const p = new Vec3(
    env.east + jitter(Math.min(80, env.east * 0.04 + 6)),
    env.north + jitter(Math.min(80, Math.abs(env.north) * 0.2 + 4)),
    env.alt + jitter(env.alt * 0.02 + 4),
  );
  const signE = rand() > 0.5 || env.east < 400 ? 1 : rand() > 0.5 ? 1 : -1;
  const v = new Vec3(
    env.ve * (env.east > 400 ? 1 : signE) + jitter(Math.abs(env.ve) * 0.08 + 1.2),
    env.vn + jitter(Math.abs(env.vn) * 0.2 + 0.8),
    env.vu + jitter(Math.abs(env.vu) * 0.08 + 1),
  );

  const up = new Vec3(0, 0, 1);
  let bodyX = up;
  if (env.tailFirst && v.len() > 20) {
    bodyX = v.neg().normalized();
  } else if (env.tilt > 0.01) {
    const az = rand() * Math.PI * 2;
    const axis = new Vec3(Math.cos(az), Math.sin(az), 0).normalized();
    bodyX = Quat.fromAxisAngle(axis, env.tilt + jitter(0.04)).rotate(up);
  }

  const q = Quat.fromRotationArc(new Vec3(1, 0, 0), bodyX);
  const omega =
    env.tilt > 0.01
      ? new Vec3(jitter(0.04), jitter(0.1), jitter(0.1))
      : new Vec3();

  return {
    p,
    v,
    q,
    omega,
    fuel: env.fuel,
    wind: env.wind,
    timeout: env.timeout,
    energy: env.energy,
  };
}

export const SAVE_VERSION = 3;
export const SAVE_KEY = "spacey-brain-v3";
