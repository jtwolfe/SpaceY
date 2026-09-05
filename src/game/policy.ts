/** 14→8→10 tanh residual net. Zero weights leave the GNC untouched. */

import {
  FIN_MAX_DEFLECT_RAD,
  GIMBAL_MAX_RAD,
  STAGE_LENGTH_M,
  START_FUEL_KG,
  THROTTLE_MAX,
  THROTTLE_MIN,
} from "./constants";
import type { Controls, Nav } from "./guidance";
import { clamp, saturate, Vec3, Quat } from "./math";

export const N_IN = 14;
export const N_HIDDEN = 8;
export const N_OUT = 10;
export const N_WEIGHTS = N_HIDDEN * N_IN + N_HIDDEN + N_OUT * N_HIDDEN + N_OUT; // 210

const OBS_POS = 400;
const OBS_VEL = 80;
const OBS_RATE = 1;

export function zeroWeights(): number[] {
  return new Array(N_WEIGHTS).fill(0);
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

export type MlpOut = { y: Float64Array; h: Float64Array };

const Y = new Float64Array(N_OUT);
const H = new Float64Array(N_HIDDEN);

export function mlpForward(w: number[], x: ArrayLike<number>): MlpOut {
  const need = N_WEIGHTS;
  Y.fill(0);
  H.fill(0);
  if (w.length < need) return { y: Y, h: H };
  const b1 = N_HIDDEN * N_IN;
  const w2 = b1 + N_HIDDEN;
  const b2 = w2 + N_OUT * N_HIDDEN;
  for (let j = 0; j < N_HIDDEN; j++) {
    let s = w[b1 + j];
    const row = j * N_IN;
    for (let i = 0; i < N_IN; i++) s += w[row + i] * x[i];
    H[j] = tanh(s);
  }
  for (let a = 0; a < N_OUT; a++) {
    let s = w[b2 + a];
    const row = w2 + a * N_HIDDEN;
    for (let j = 0; j < N_HIDDEN; j++) s += w[row + j] * H[j];
    Y[a] = tanh(s);
  }
  return { y: Y, h: H };
}

export function observe(nav: Nav, bodyY: Vec3, omega: Vec3): number[] {
  const bx = nav.bodyX;
  let bz = bx.cross(bodyY);
  if (bz.len() < 1e-8) bz = new Vec3(0, 1, 0);
  else bz = bz.normalized();
  let by = bz.cross(bx);
  if (by.len() < 1e-8) by = new Vec3(0, 1, 0);
  else by = by.normalized();
  const q = Quat.fromAxes(bx, by, bz).hemisphere();
  const engine = nav.p.add(bx.scale(-STAGE_LENGTH_M * 0.5));
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
  ];
}

/** Nudge GNC actuators. |y|=1 is a small, recoverable offset — not a takeover. */
export function applyResidual(u: Controls, y: ArrayLike<number>) {
  u.throttle = saturate(u.throttle + y[0] * 0.22);
  if (u.throttle > 0.02 && u.throttle < THROTTLE_MIN) u.throttle = THROTTLE_MIN;
  if (u.throttle > THROTTLE_MAX) u.throttle = THROTTLE_MAX;
  u.gimbalY = clamp(u.gimbalY + y[1] * GIMBAL_MAX_RAD * 0.7 + y[7] * GIMBAL_MAX_RAD * 0.2, -GIMBAL_MAX_RAD, GIMBAL_MAX_RAD);
  u.gimbalZ = clamp(u.gimbalZ + y[2] * GIMBAL_MAX_RAD * 0.7 + y[8] * GIMBAL_MAX_RAD * 0.2, -GIMBAL_MAX_RAD, GIMBAL_MAX_RAD);
  u.finPitch = clamp(u.finPitch + y[3] * FIN_MAX_DEFLECT_RAD * 0.7, -FIN_MAX_DEFLECT_RAD, FIN_MAX_DEFLECT_RAD);
  u.finYaw = clamp(u.finYaw + y[4] * FIN_MAX_DEFLECT_RAD * 0.7, -FIN_MAX_DEFLECT_RAD, FIN_MAX_DEFLECT_RAD);
  u.finRoll = clamp(u.finRoll + y[5] * FIN_MAX_DEFLECT_RAD * 0.7 + y[9] * FIN_MAX_DEFLECT_RAD * 0.3, -FIN_MAX_DEFLECT_RAD, FIN_MAX_DEFLECT_RAD);
  if (y[6] > 0.42) u.nEngines = 3;
  else if (y[6] < -0.42 && u.nEngines > 0) u.nEngines = 1;
}
