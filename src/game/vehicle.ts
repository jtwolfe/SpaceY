import {
  AOA_DESTROY_RAD,
  BODY_CP_X_M,
  ENGINE_MIN_BURN_S,
  ENGINE_RESTART_DELAY_S,
  FIN_AREA_M2,
  FIN_ARM_M,
  FIN_CD0,
  FIN_CL_DELTA,
  FIN_MAX_DEFLECT_RAD,
  FIN_Q_FADE_PA,
  FIN_Q_FULL_PA,
  G0,
  G_DESTROY,
  GIMBAL_MAX_RAD,
  MERLIN_ISP_SL_S,
  MERLIN_ISP_VAC_S,
  MERLIN_THRUST_SL_N,
  MERLIN_THRUST_VAC_N,
  Q_ALPHA_DESTROY,
  Q_DESTROY_PA,
  Q_FOR_AOA_DESTROY_PA,
  RATE_DESTROY_RAD_S,
  RCS_ANG_ACCEL,
  RCS_Q_HANDOFF_PA,
  REF_AREA_M2,
  STAGE_LENGTH_M,
  STAGE_RADIUS_M,
  THROTTLE_MAX,
  THROTTLE_MIN,
} from "./constants";
import type { Air } from "./atmosphere";
import { angleBetween, clamp, lerp, Vec3 } from "./math";

export type Aero = {
  forceBody: Vec3;
  momentBody: Vec3;
  mach: number;
  q: number;
  aoa: number;
  cd: number;
  heat: number;
  finDelta: [number, number, number, number];
};

export function mixFins(pitch: number, yaw: number, roll: number): [number, number, number, number] {
  const max = FIN_MAX_DEFLECT_RAD;
  const d: [number, number, number, number] = [0, 0, 0, 0];
  for (let i = 0; i < 4; i++) {
    const th = Math.PI / 4 + i * (Math.PI / 2);
    const divert = clamp(pitch * Math.cos(th) + yaw * Math.sin(th), -max, max);
    const head = max - Math.abs(divert);
    d[i] = divert + clamp(roll, -head, head);
  }
  return d;
}

export function finQEnable(q: number) {
  if (q <= FIN_Q_FADE_PA) return 0;
  if (q >= FIN_Q_FULL_PA) return 1;
  return (q - FIN_Q_FADE_PA) / (FIN_Q_FULL_PA - FIN_Q_FADE_PA);
}

export function cd0(mach: number) {
  if (mach < 0.8) return 0.82;
  if (mach < 1.25) return 0.82 + (0.55 * (mach - 0.8)) / 0.45;
  if (mach < 5) return 1.37 - (0.42 * (mach - 1.25)) / 3.75;
  return 0.95;
}

export function aero(vRelBody: Vec3, air: Air, finPitch: number, finYaw: number, finRoll: number): Aero {
  const v = vRelBody.len();
  const q = 0.5 * air.density * v * v;
  const mach = air.speedOfSound > 1 ? v / air.speedOfSound : 0;
  if (v < 0.5) {
    return {
      forceBody: new Vec3(),
      momentBody: new Vec3(),
      mach,
      q,
      aoa: 0,
      cd: cd0(mach),
      heat: 0,
      finDelta: [0, 0, 0, 0],
    };
  }
  const vhat = vRelBody.scale(1 / v);
  const tail = new Vec3(-1, 0, 0);
  const aoa = angleBetween(tail, vhat);
  const sAoa = Math.sin(aoa);
  const cl = 0.55 * Math.sin(2 * aoa);
  const cdBody = cd0(mach) + 1.8 * sAoa * sAoa;
  const fDrag = vhat.scale(-q * REF_AREA_M2 * cdBody);
  const side = tail.cross(vhat);
  const liftDir =
    side.len() < 1e-8 ? new Vec3() : vhat.cross(side.normalized()).normalized();
  const fLift = liftDir.scale(q * REF_AREA_M2 * cl);
  const fBody = fDrag.add(fLift);
  const rBody = new Vec3(BODY_CP_X_M, 0, 0);

  const fp = clamp(finPitch, -FIN_MAX_DEFLECT_RAD, FIN_MAX_DEFLECT_RAD);
  const fy = clamp(finYaw, -FIN_MAX_DEFLECT_RAD, FIN_MAX_DEFLECT_RAD);
  const fr = clamp(finRoll, -FIN_MAX_DEFLECT_RAD, FIN_MAX_DEFLECT_RAD);
  const delta = mixFins(fp, fy, fr);

  let fFins = new Vec3();
  let mFins = new Vec3();
  let cdLat = 0;
  for (let i = 0; i < 4; i++) {
    const th = Math.PI / 4 + i * (Math.PI / 2);
    const cy = Math.cos(th);
    const sy = Math.sin(th);
    const rFin = new Vec3(FIN_ARM_M, STAGE_RADIUS_M * cy, STAGE_RADIUS_M * sy);
    const d = delta[i];
    const cdI = FIN_CD0 + 0.85 * Math.abs(d) + 0.3 * sAoa * sAoa;
    cdLat += (cdI * FIN_AREA_M2) / REF_AREA_M2;
    const fAx = vhat.scale(-q * FIN_AREA_M2 * cdI);
    const tangent = new Vec3(0, -sy, cy);
    const fCmd = tangent.scale(q * FIN_AREA_M2 * FIN_CL_DELTA * d);
    const fi = fAx.add(fCmd);
    fFins = fFins.add(fi);
    mFins = mFins.add(rFin.cross(fi));
  }

  const heat = 1.83e-4 * v * v * v * Math.sqrt(Math.max(0, air.density));
  const f = fBody.add(fFins);
  const axial = Math.max(0, -f.dot(vhat));
  const cd = q * REF_AREA_M2 > 1e-6 ? axial / (q * REF_AREA_M2) : cdBody + cdLat;

  return {
    forceBody: f,
    momentBody: rBody.cross(fBody).add(mFins),
    mach,
    q,
    aoa,
    cd,
    heat,
    finDelta: delta,
  };
}

export type Burn = {
  forceBody: Vec3;
  momentBody: Vec3;
  mdot: number;
  thrust: number;
};

export function propulsion(
  throttleCmd: number,
  gimbalY: number,
  gimbalZ: number,
  nEngines: number,
  pressurePa: number,
  fuel: number,
): Burn {
  if (nEngines === 0 || fuel <= 1 || throttleCmd <= 0.02) {
    return { forceBody: new Vec3(), momentBody: new Vec3(), mdot: 0, thrust: 0 };
  }
  const throttle = clamp(throttleCmd, THROTTLE_MIN, THROTTLE_MAX);
  const pFrac = clamp(pressurePa / 101_325, 0, 1);
  const thrustOne = lerp(MERLIN_THRUST_VAC_N, MERLIN_THRUST_SL_N, pFrac);
  const isp = lerp(MERLIN_ISP_VAC_S, MERLIN_ISP_SL_S, pFrac);
  const gy = clamp(gimbalY, -GIMBAL_MAX_RAD, GIMBAL_MAX_RAD);
  const gz = clamp(gimbalZ, -GIMBAL_MAX_RAD, GIMBAL_MAX_RAD);
  const dir = new Vec3(
    Math.cos(gy) * Math.cos(gz),
    Math.sin(gz),
    Math.sin(gy) * Math.cos(gz),
  ).normalized();
  const thrust = thrustOne * nEngines * throttle;
  const force = dir.scale(thrust);
  const rEng = new Vec3(-STAGE_LENGTH_M * 0.5, 0, 0);
  return {
    forceBody: force,
    momentBody: rEng.cross(force),
    mdot: thrust / (isp * G0),
    thrust,
  };
}

export class EngineGate {
  on = false;
  lights = 0;
  relights = 0;
  lastLightT = -1e9;
  lastShutdownT = -1e9;
  holdThrottle = 0;
  holdN = 0;

  apply(t: number, throttle: number, nEngines: number): { throttle: number; n: number } {
    const want = nEngines > 0 && throttle > 0.02;
    if (this.on) {
      if (want) {
        this.holdThrottle = Math.max(throttle, THROTTLE_MIN);
        this.holdN = Math.max(nEngines, 1);
        return { throttle: this.holdThrottle, n: this.holdN };
      }
      if (t - this.lastLightT < ENGINE_MIN_BURN_S) {
        return { throttle: Math.max(this.holdThrottle, THROTTLE_MIN), n: Math.max(this.holdN, 1) };
      }
      this.on = false;
      this.lastShutdownT = t;
      this.holdThrottle = 0;
      this.holdN = 0;
      return { throttle: 0, n: 0 };
    }
    if (want) {
      if (this.lights > 0 && t - this.lastShutdownT < ENGINE_RESTART_DELAY_S) {
        return { throttle: 0, n: 0 };
      }
      this.on = true;
      this.lastLightT = t;
      this.lights += 1;
      if (this.lights > 1) this.relights += 1;
      this.holdThrottle = Math.max(throttle, THROTTLE_MIN);
      this.holdN = Math.max(nEngines, 1);
      return { throttle: this.holdThrottle, n: this.holdN };
    }
    return { throttle: 0, n: 0 };
  }
}

export type DestroyReason =
  | "none"
  | "max-Q"
  | "over-G"
  | "q-alpha"
  | "AoA"
  | "spin"
  | "impact";

export function checkDestruction(
  q: number,
  aoa: number,
  accelG: number,
  rate: number,
  enabled: boolean,
): DestroyReason {
  if (!enabled) return "none";
  if (q > Q_DESTROY_PA) return "max-Q";
  if (accelG > G_DESTROY) return "over-G";
  if ((q / 1000) * (Math.abs(aoa) * 180) / Math.PI > Q_ALPHA_DESTROY) return "q-alpha";
  if (Math.abs(aoa) > AOA_DESTROY_RAD && q > Q_FOR_AOA_DESTROY_PA) return "AoA";
  if (rate > RATE_DESTROY_RAD_S) return "spin";
  return "none";
}

export function rcsMoment(omega: Vec3, errBody: Vec3, inertia: { x: number; y: number; z: number }, q: number) {
  const blend = clamp(1 - q / RCS_Q_HANDOFF_PA, 0, 1);
  if (blend < 1e-4) return new Vec3();
  const wmax = 0.42;
  const wCmdY = clamp(2.2 * errBody.y, -wmax, wmax);
  const wCmdZ = clamp(2.2 * errBody.z, -wmax, wmax);
  const ay = clamp((wCmdY - omega.y) * 3, -RCS_ANG_ACCEL, RCS_ANG_ACCEL);
  const az = clamp((wCmdZ - omega.z) * 3, -RCS_ANG_ACCEL, RCS_ANG_ACCEL);
  const ax = clamp(-omega.x * 2.5, -RCS_ANG_ACCEL, RCS_ANG_ACCEL);
  return new Vec3(inertia.x * ax, inertia.y * ay, inertia.z * az).scale(blend);
}

export function finQEnableExport(q: number) {
  return finQEnable(q);
}
