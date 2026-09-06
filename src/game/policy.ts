/** Growable residual MLP. Starts 21→8→10; extra hidden blocks are identity at zero. */

import {
  FIN_MAX_DEFLECT_RAD,
  GIMBAL_MAX_RAD,
  GEAR_ENGINE_ALT_M,
  N_ENGINES_LANDING,
  STAGE_LENGTH_M,
  START_FUEL_KG,
  THROTTLE_MAX,
  THROTTLE_MIN,
} from "./constants";
import type { Controls, Nav, Phase } from "./guidance";
import { clamp, saturate, Vec3, Quat } from "./math";

export const N_IN = 21;
export const N_HIDDEN = 8;
export const N_HIDDEN_LAYERS_MIN = 1;
/** Glide plateaus at 3 and used to sit at 4/4 with no RTLS block left. */
export const N_HIDDEN_LAYERS_MAX = 6;
export const N_HIDDEN_LAYERS = N_HIDDEN_LAYERS_MAX;
export const N_OUT = 10;

export function nWeights(layers: number) {
  const L = Math.min(N_HIDDEN_LAYERS_MAX, Math.max(N_HIDDEN_LAYERS_MIN, Math.round(layers)));
  return N_HIDDEN * N_IN + N_HIDDEN + (L - 1) * (N_HIDDEN * N_HIDDEN + N_HIDDEN) + N_OUT * N_HIDDEN + N_OUT;
}

export function layersFromLen(len: number) {
  for (let L = N_HIDDEN_LAYERS_MIN; L <= N_HIDDEN_LAYERS_MAX; L++) if (nWeights(L) === len) return L;
  return 0;
}

export function topologyOf(layers: number) {
  const L = Math.min(N_HIDDEN_LAYERS_MAX, Math.max(N_HIDDEN_LAYERS_MIN, Math.round(layers)));
  return `${N_IN}→${Array.from({ length: L }, () => N_HIDDEN).join("→")}→${N_OUT}`;
}

export const N_WEIGHTS = nWeights(N_HIDDEN_LAYERS_MIN);
export const N_WEIGHTS_MAX = nWeights(N_HIDDEN_LAYERS_MAX);
export const TOPOLOGY = topologyOf(N_HIDDEN_LAYERS_MIN);

const BLOCK = N_HIDDEN * N_HIDDEN + N_HIDDEN;
const OUT_N = N_OUT * N_HIDDEN + N_OUT;

export function zeroWeights(layers = N_HIDDEN_LAYERS_MIN): number[] {
  return new Array(nWeights(layers)).fill(0);
}

/** Insert a zero residual block before the output. Forward is unchanged at init. */
export function growWeights(w: number[]): number[] {
  const L = layersFromLen(w.length);
  if (L < 1 || L >= N_HIDDEN_LAYERS_MAX) return w;
  const head = w.slice(0, w.length - OUT_N);
  const tail = w.slice(w.length - OUT_N);
  return [...head, ...new Array(BLOCK).fill(0), ...tail];
}

export function growVec(v: number[], fill: number, fromLen: number): number[] {
  if (v.length !== fromLen) return new Array(nWeights(layersFromLen(fromLen) + 1 || N_HIDDEN_LAYERS_MIN)).fill(fill);
  const L = layersFromLen(fromLen);
  if (L < 1 || L >= N_HIDDEN_LAYERS_MAX) return v.slice();
  const head = v.slice(0, fromLen - OUT_N);
  const tail = v.slice(fromLen - OUT_N);
  return [...head, ...new Array(BLOCK).fill(fill), ...tail];
}

function tanh(x: number) {
  if (x > 8) return 1;
  if (x < -8) return -1;
  const e = Math.exp(2 * x);
  return (e - 1) / (e + 1);
}

function asinh(x: number) {
  return Math.asinh(x);
}

export type MlpOut = { y: Float64Array; h: Float64Array; layers: number };

const Y = new Float64Array(N_OUT);
const H = new Float64Array(N_HIDDEN * N_HIDDEN_LAYERS_MAX);

export function mlpForward(w: number[], x: ArrayLike<number>): MlpOut {
  Y.fill(0);
  H.fill(0);
  const L = layersFromLen(w.length);
  if (L < 1) return { y: Y, h: H.subarray(0, N_HIDDEN), layers: 0 };
  for (let j = 0; j < N_HIDDEN; j++) {
    let s = w[N_HIDDEN * N_IN + j];
    const row = j * N_IN;
    for (let i = 0; i < N_IN; i++) s += w[row + i] * x[i];
    H[j] = tanh(s);
  }
  let off = N_HIDDEN * N_IN + N_HIDDEN;
  for (let layer = 1; layer < L; layer++) {
    const prev = (layer - 1) * N_HIDDEN;
    const cur = layer * N_HIDDEN;
    const wOff = off;
    const bOff = off + N_HIDDEN * N_HIDDEN;
    for (let j = 0; j < N_HIDDEN; j++) {
      let s = w[bOff + j];
      const row = wOff + j * N_HIDDEN;
      for (let i = 0; i < N_HIDDEN; i++) s += w[row + i] * H[prev + i];
      H[cur + j] = H[prev + j] + tanh(s);
    }
    off = bOff + N_HIDDEN;
  }
  const last = (L - 1) * N_HIDDEN;
  const wOut = off;
  const bOut = off + N_OUT * N_HIDDEN;
  for (let a = 0; a < N_OUT; a++) {
    let s = w[bOut + a];
    const row = wOut + a * N_HIDDEN;
    for (let j = 0; j < N_HIDDEN; j++) s += w[row + j] * H[last + j];
    Y[a] = tanh(s);
  }
  return { y: Y, h: H.subarray(0, L * N_HIDDEN), layers: L };
}

const OBS_POS = 400;
const OBS_VEL = 80;
const OBS_RATE = 1;

export type RefObs = { dp: Vec3; dv: Vec3; tLight: number };

export function observe(nav: Nav, bodyY: Vec3, omega: Vec3, ref?: RefObs | null): number[] {
  const bx = nav.bodyX;
  let bz = bx.cross(bodyY);
  if (bz.len() < 1e-8) bz = new Vec3(0, 1, 0);
  else bz = bz.normalized();
  let by = bz.cross(bx);
  if (by.len() < 1e-8) by = new Vec3(0, 1, 0);
  else by = by.normalized();
  const q = Quat.fromAxes(bx, by, bz).hemisphere();
  const engine = nav.p.add(bx.scale(-STAGE_LENGTH_M * 0.5));
  const dp = ref?.dp ?? Vec3.ZERO;
  const dv = ref?.dv ?? Vec3.ZERO;
  const tLight = ref?.tLight ?? 0;
  return [
    asinh(engine.x / OBS_POS),
    asinh(engine.y / OBS_POS),
    asinh(nav.engineAlt / OBS_POS),
    asinh(nav.v.x / OBS_VEL),
    asinh(nav.v.y / OBS_VEL),
    asinh(nav.v.z / OBS_VEL),
    q.w,
    q.x,
    q.y,
    q.z,
    asinh(omega.x / OBS_RATE),
    asinh(omega.y / OBS_RATE),
    asinh(omega.z / OBS_RATE),
    clamp(nav.fuel / START_FUEL_KG, 0, 2),
    asinh(dp.x / OBS_POS),
    asinh(dp.y / OBS_POS),
    asinh(dp.z / OBS_POS),
    asinh(dv.x / OBS_VEL),
    asinh(dv.y / OBS_VEL),
    asinh(dv.z / OBS_VEL),
    clamp(tLight / 12, -1, 1),
  ];
}

/** Glide-tuned residual at 80 km / 2 km/s fights entry GNC. Freeze until the
 *  booster is in the same regime the earlier rungs trained on. */
export function residualLive(energy: number, phase: Phase, alt: number, speed: number, entryOn: boolean) {
  if (energy < 0.85) return true;
  if (entryOn) return false;
  if (phase === "exo" || phase === "entry") return false;
  if (phase === "glide" && (alt > 12_000 || speed > 450)) return false;
  return true;
}

/** Nudge GNC actuators. |y|=1 is a small, recoverable offset — not a takeover.
 *  Residual cannot relight a shut engine (0.22 of throttle would snap to Merlin min 40% and hover).
 *  lockEngines: RTLS landing keeps GNC's 1-Merlin (y[6] must not dump a 3-wide). */
export function applyResidual(u: Controls, y: ArrayLike<number>, lockEngines = false) {
  const gncOff = u.nEngines <= 0 || u.throttle <= 0.02;
  u.gimbalY = clamp(u.gimbalY + y[1] * GIMBAL_MAX_RAD * 0.7 + y[7] * GIMBAL_MAX_RAD * 0.2, -GIMBAL_MAX_RAD, GIMBAL_MAX_RAD);
  u.gimbalZ = clamp(u.gimbalZ + y[2] * GIMBAL_MAX_RAD * 0.7 + y[8] * GIMBAL_MAX_RAD * 0.2, -GIMBAL_MAX_RAD, GIMBAL_MAX_RAD);
  u.finPitch = clamp(u.finPitch + y[3] * FIN_MAX_DEFLECT_RAD * 0.7, -FIN_MAX_DEFLECT_RAD, FIN_MAX_DEFLECT_RAD);
  u.finYaw = clamp(u.finYaw + y[4] * FIN_MAX_DEFLECT_RAD * 0.7, -FIN_MAX_DEFLECT_RAD, FIN_MAX_DEFLECT_RAD);
  u.finRoll = clamp(u.finRoll + y[5] * FIN_MAX_DEFLECT_RAD * 0.7 + y[9] * FIN_MAX_DEFLECT_RAD * 0.3, -FIN_MAX_DEFLECT_RAD, FIN_MAX_DEFLECT_RAD);
  if (gncOff) {
    u.throttle = 0;
    u.nEngines = 0;
    return;
  }
  u.throttle = saturate(u.throttle + y[0] * 0.22);
  if (u.throttle > 0.02 && u.throttle < THROTTLE_MIN) u.throttle = THROTTLE_MIN;
  if (u.throttle > THROTTLE_MAX) u.throttle = THROTTLE_MAX;
  if (u.throttle <= 0.02) {
    u.throttle = 0;
    u.nEngines = 0;
    return;
  }
  if (lockEngines) return;
  if (y[6] > 0.42) u.nEngines = 3;
  else if (y[6] < -0.42 && u.nEngines > 0) u.nEngines = 1;
}

/** Re-apply after the net: a landing-envelope climb cannot keep thrusting hard.
 *  Above gear, floor to 1 @ min and stay lit — dumping mid-air is a drop. */
export function keepSinking(u: Controls, nav: Nav) {
  if (nav.engineAlt > 8_000) return;
  if (u.nEngines <= 0 || u.throttle <= 0.02) return;
  if (nav.v.z <= 0.8) return;
  if (nav.engineAlt > GEAR_ENGINE_ALT_M) {
    u.nEngines = N_ENGINES_LANDING;
    u.throttle = THROTTLE_MIN;
    return;
  }
  u.throttle = 0;
  u.nEngines = 0;
}
