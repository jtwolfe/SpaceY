import {
  FIN_MAX_DEFLECT_RAD,
  G0,
  GIMBAL_MAX_RAD,
  MERLIN_THRUST_SL_N,
  N_ENGINES_ENTRY,
  N_ENGINES_LANDING,
  STAGE_LENGTH_M,
  SUCCESS_ENGINE_ALT_M,
  SUCCESS_HVEL_MPS,
  SUCCESS_PAD_OFFSET_M,
  SUCCESS_SPEED_MPS,
  SUCCESS_TILT_RAD,
  SUICIDE_FUEL_KG,
  THROTTLE_MIN,
} from "./constants";
import { clamp, saturate, Vec3 } from "./math";

export type Phase = "exo" | "entry" | "glide" | "landing";

export type TermReason =
  | "none"
  | "landed"
  | "miss"
  | "destroyed"
  | "timeout"
  | "fuel"
  | "corridor";

export type Controls = {
  throttle: number;
  gimbalY: number;
  gimbalZ: number;
  finPitch: number;
  finYaw: number;
  finRoll: number;
  nEngines: number;
};

export function emptyControls(): Controls {
  return {
    throttle: 0,
    gimbalY: 0,
    gimbalZ: 0,
    finPitch: 0,
    finYaw: 0,
    finRoll: 0,
    nEngines: 0,
  };
}

export type Nav = {
  alt: number;
  engineAlt: number;
  speed: number;
  v: Vec3;
  p: Vec3;
  rangeH: number;
  tilt: number;
  q: number;
  aoa: number;
  mach: number;
  fuel: number;
  mass: number;
  bodyX: Vec3;
};

export type Gains = number[];
/** ignite, kp, kd, maxTilt, entry, dive, att, tvc, fin, three, thr, yaw */
export const N_GAINS = 12;
export const NOMINAL_GAINS: Gains = Array.from({ length: N_GAINS }, () => 1);

export function classifyPhase(nav: Nav, landingLatched: boolean): Phase {
  if (landingLatched) return "landing";
  if (shouldStartLanding(nav)) return "landing";
  if (nav.alt > 78_000 && nav.q < 80 && nav.speed < 400) return "exo";
  const pred = predictedLandingRange(nav);
  const long = pred > nav.rangeH + 8_000 && nav.speed > 450 && nav.alt > 12_000;
  if (long) return "entry";
  if (nav.rangeH < 5_000 && nav.speed < 520 && !long) {
    if (nav.alt > 8_000) return "glide";
    return "landing";
  }
  if (nav.alt > 12_000 && nav.speed > vRef(nav.alt) + 40) return "entry";
  if (nav.speed > 1_550 && nav.alt < 105_000) return "entry";
  if (nav.alt > 8_000) return "glide";
  return "landing";
}

export function vRef(alt: number) {
  if (alt > 70_000) return 1_800;
  if (alt > 45_000) return 550 + (1_000 * (alt - 45_000)) / 25_000;
  if (alt > 20_000) return 280 + (270 * (alt - 20_000)) / 25_000;
  if (alt > 8_000) return 120 + (160 * (alt - 8_000)) / 12_000;
  return 50 + (70 * alt) / 8_000;
}

export function shouldStartLanding(nav: Nav) {
  if (nav.rangeH > 8_000) return false;
  if (nav.alt > 12_000 && nav.rangeH > 4_000) return false;
  const tSl = MERLIN_THRUST_SL_N * N_ENGINES_LANDING;
  const aUp = Math.max(2, (tSl * 0.85) / nav.mass - G0);
  const vDown = Math.max(0, -nav.v.z);
  const sBurn = (vDown * vDown) / (2 * aUp) + 40;
  return (
    nav.engineAlt < sBurn ||
    (nav.engineAlt < 2_400 && vDown > 40 && nav.rangeH < 3_000)
  );
}

function predictedLandingRange(nav: Nav) {
  const vh = Math.hypot(nav.v.x, nav.v.y);
  const vd = Math.max(15, -nav.v.z);
  const t = nav.alt / vd;
  const drag = Math.min(3.5, 1 + nav.q / 50_000);
  return (vh * t * 0.55) / drag;
}

function g(gains: Gains, i: number) {
  return gains[i] ?? 1;
}

function restartFeasible(pz: number, vDown: number, mass: number) {
  const t = 6.2;
  const z2 = pz - vDown * t - 0.5 * G0 * t * t;
  const v2 = vDown + G0 * t;
  const a1 = clamp(MERLIN_THRUST_SL_N / mass - G0, 4, 40);
  const s2 = (v2 * v2) / (2 * a1) + 8;
  return z2 > Math.max(s2, 140) + 50;
}

export function nominalControls(
  nav: Nav,
  phase: Phase,
  engineOn: boolean,
  gains: Gains,
): { u: Controls; desiredX: Vec3 } {
  let desiredX = new Vec3(0, 0, 1);
  const u = emptyControls();

  if (phase === "exo" || phase === "entry") {
    if (nav.v.len() > 10) desiredX = aimRetro(nav, gains);
    const reserve = SUICIDE_FUEL_KG * 1.05;
    const pred = predictedLandingRange(nav);
    const closing =
      nav.rangeH < 1 || -nav.p.x * nav.v.x + -nav.p.y * nav.v.y > 0;
    const overshoot = closing ? pred - nav.rangeH : pred + nav.rangeH;
    const vTarget = vRef(nav.alt);
    const tooFast = nav.speed > vTarget + 80;
    const qHot = nav.q > 22_000 && nav.speed > 400;
    const long = overshoot > 6_000 && nav.speed > 500;
    const hypersonic = nav.speed > 1_550;
    const fuelOk = nav.fuel > reserve;
    if (fuelOk && (hypersonic || tooFast || qHot || long) && nav.alt < 95_000) {
      u.nEngines = N_ENGINES_ENTRY;
      const need = hypersonic
        ? nav.speed - 1_400
        : tooFast
          ? nav.speed - vTarget
          : qHot
            ? nav.speed - 380
            : overshoot / 12;
      let thr = saturate(0.4 + (need / 500) * g(gains, 4));
      const qG = (nav.q * 10.56) / nav.mass / G0;
      if (qG > 6.3) thr *= clamp(9 / Math.max(1, qG), 0.4, 1);
      u.throttle = saturate(thr);
    }
  } else if (phase === "glide") {
    const range = Math.max(1, Math.hypot(nav.p.x, nav.p.y));
    if (range < 2_500 && nav.speed < 420) {
      desiredX =
        nav.v.len() > 20 ? nav.v.neg().normalized() : new Vec3(0, 0, 1);
      u.nEngines = 0;
    } else {
      const pred = predictedLandingRange(nav);
      const energyErr = ((pred - range) / 8_000) * g(gains, 5);
      const dive = clamp(-energyErr, -0.2, 0.2);
      const vdir =
        nav.v.len() > 5
          ? nav.v.normalized()
          : new Vec3(-nav.p.x / range, -nav.p.y / range, -0.4).normalized();
      const aim = new Vec3(vdir.x, vdir.y, clamp(vdir.z + dive, -0.98, -0.12)).normalized();
      desiredX = aimRetroDir(nav, aim.neg(), gains);
      u.nEngines = 0;
    }
  } else {
    hoverSlam(nav, u, (x) => {
      desiredX = x;
    }, engineOn, gains);
  }

  return { u, desiredX };
}

function hoverSlam(
  nav: Nav,
  u: Controls,
  setDesired: (x: Vec3) => void,
  engineOn: boolean,
  gains: Gains,
) {
  const pz = Math.max(0.5, nav.engineAlt);
  const vDown = Math.max(0, -nav.v.z);
  const climbing = nav.v.z > 3;
  const vLand = 6.0;
  const hLand = 4.0;
  const a1 = clamp(MERLIN_THRUST_SL_N / nav.mass - G0, 4, 40);
  const a3 = clamp((MERLIN_THRUST_SL_N * N_ENGINES_ENTRY) / nav.mass - G0, 8, 90);
  const ign = g(gains, 0);
  const s1 = Math.max(0, vDown * vDown - vLand * vLand) / (2 * a1 * 0.96) + hLand;
  const s3 = Math.max(0, vDown * vDown - vLand * vLand) / (2 * a3) + 8;

  const vh = Math.hypot(nav.v.x, nav.v.y);
  const range = nav.rangeH;
  const overPad = range < 22 && vh < 5.5;
  const use3 = vDown > 90 && s1 > pz + 20 && pz > 110;
  const sLight = (use3 ? s3 + 8 : s1) * ign;
  const divert = !overPad && pz > 160 && !use3;

  if (climbing && overPad && pz > 80) {
    u.throttle = 0;
    u.nEngines = 0;
    setDesired(new Vec3(0, 0, 1));
    return;
  }

  if (divert) {
    u.nEngines = N_ENGINES_LANDING;
    const vHold = clamp(28 + pz * 0.02, 26, 80);
    if (climbing || vDown < vHold) u.throttle = THROTTLE_MIN;
    else if (range > 40) u.throttle = saturate(0.72 * g(gains, 10));
    else u.throttle = 0.55;
    u.throttle = Math.max(THROTTLE_MIN, u.throttle);
  } else if (overPad && engineOn && pz > sLight + 160 && restartFeasible(pz, vDown, nav.mass)) {
    u.throttle = 0;
    u.nEngines = 0;
  } else if (!engineOn && pz > sLight + 8 && pz > hLand + 10) {
    u.throttle = 0;
    u.nEngines = 0;
  } else {
    u.nEngines = use3 ? N_ENGINES_ENTRY : N_ENGINES_LANDING;
    let thr = saturate(0.9 * g(gains, 10));
    if (pz < 18 && vDown > 4) thr = 1;
    u.throttle = Math.max(THROTTLE_MIN, thr);
  }

  if (engineOn && overPad && pz < 70 && vDown > 0.8 && !climbing) {
    u.nEngines = Math.max(u.nEngines, N_ENGINES_LANDING);
    u.throttle = Math.max(u.throttle, 0.85);
  }

  const thrusting = u.nEngines > 0 && u.throttle > 0.02;
  let azCmd = thrusting
    ? Math.max(2, (u.throttle * MERLIN_THRUST_SL_N * u.nEngines) / nav.mass)
    : 2;
  const ahMax = Math.max(2, azCmd * Math.sin(0.5));
  const sBrake = (vh * vh) / (2 * ahMax) + 20;
  const closing = range < 1 || nav.p.x * nav.v.x + nav.p.y * nav.v.y < 0;

  let kp = (pz < 80 ? 0.32 : 0.18) * g(gains, 1);
  let kd = (pz < 80 ? 2.6 : 2.0) * g(gains, 2);
  if (divert && closing && range < sBrake) {
    kp *= 0.35;
    kd *= 1.55;
  }

  const ax = -kp * nav.p.x - kd * nav.v.x;
  const ay = -kp * nav.p.y - kd * nav.v.y;
  const horiz = Math.hypot(ax, ay);
  const maxTilt =
    (pz < 28 ? 0.04 : pz < 70 ? 0.09 : pz < 200 ? 0.2 : divert ? 0.52 : 0.28) * g(gains, 3);

  if (!thrusting) {
    let desired = nav.v.len() > 18 ? nav.v.neg().normalized() : new Vec3(0, 0, 1);
    if (range > 18) {
      const pull = clamp(range / 2_200, 0.05, 0.3);
      const hdir = new Vec3(-nav.p.x, -nav.p.y, 0);
      if (hdir.len() > 1) desired = desired.add(hdir.normalized().scale(pull)).normalized();
    }
    setDesired(desired);
    return;
  }

  let tilt0 = Math.min(maxTilt, horiz / Math.max(2, azCmd));
  let tilt = tilt0;

  if (divert && !overPad && pz > sLight + 40) {
    const vert = azCmd * Math.cos(Math.max(0.05, tilt0));
    if (vert > G0 * 0.9 || climbing || vDown < 24) {
      const maxAz = (G0 * 0.86) / Math.max(0.25, Math.cos(Math.max(tilt0, 0.2)));
      u.throttle = clamp(
        (maxAz * nav.mass) / (MERLIN_THRUST_SL_N * Math.max(1, u.nEngines)),
        THROTTLE_MIN,
        u.throttle,
      );
      azCmd = (u.throttle * MERLIN_THRUST_SL_N * u.nEngines) / nav.mass;
      if (azCmd * Math.cos(Math.max(0.05, tilt0)) > G0 * 0.9) {
        const cosMax = clamp((G0 * 0.86) / Math.max(2, azCmd), 0.2, 0.96);
        tilt = Math.max(tilt0, Math.acos(cosMax));
      }
    }
  }

  tilt = Math.min(tilt, 0.72);

  if (horiz > 1e-4 || tilt > 0.05) {
    let hdir = horiz > 1e-4 ? new Vec3(ax, ay, 0).normalized() : new Vec3(-nav.p.x, -nav.p.y, 0);
    if (hdir.len() < 1e-6) {
      const vv = new Vec3(nav.v.x, nav.v.y, 0);
      hdir = vv.len() > 0.5 ? vv.neg().normalized() : new Vec3(1, 0, 0);
    } else hdir = hdir.normalized();
    setDesired(
      new Vec3(0, 0, 1)
        .scale(Math.cos(tilt))
        .add(hdir.scale(Math.sin(tilt)))
        .normalized(),
    );
  } else {
    setDesired(new Vec3(0, 0, 1));
  }
}

function aimRetro(nav: Nav, gains: Gains) {
  return aimRetroDir(nav, nav.v.neg().normalized(), gains);
}

function aimRetroDir(nav: Nav, retroEnu: Vec3, gains: Gains) {
  let dir = retroEnu.clone();
  const los = new Vec3(-nav.p.x, -nav.p.y, 0);
  const r = los.len();
  if (r > 80) {
    const losH = los.scale(1 / r);
    const vh = new Vec3(nav.v.x, nav.v.y, 0);
    const vn = vh.len();
    const sinErr = vn > 15 ? (vh.x * losH.y - vh.y * losH.x) / vn : 0;
    const k = (nav.alt > 50_000 ? 0.9 : 1.1) * g(gains, 11);
    const yaw = clamp(sinErr * k, -0.55, 0.55);
    dir = new Vec3(dir.x - losH.y * yaw, dir.y + losH.x * yaw, dir.z);
    const n = dir.len();
    if (n > 1e-8) dir = dir.scale(1 / n);
  }
  return dir;
}

export function attitudeCommand(
  bodyX: Vec3,
  bodyY: Vec3,
  bodyZ: Vec3,
  omega: Vec3,
  desiredX: Vec3,
  phase: Phase,
  qDyn: number,
  gains: Gains,
): { gimY: number; gimZ: number; finP: number; finY: number; finR: number; errBody: Vec3 } {
  const err = bodyX.cross(desiredX.normalized());
  const errBody = new Vec3(err.dot(bodyX), err.dot(bodyY), err.dot(bodyZ));
  let wmax = 0.12;
  if (phase === "landing" && qDyn < 12_000) wmax = 0.35;
  else if (phase === "glide" && qDyn < 110_000) wmax = 0.32;
  else if (phase === "landing" && qDyn < 40_000) wmax = 0.28;
  wmax *= g(gains, 6);
  const wCmdY = clamp(1.8 * errBody.y, -wmax, wmax);
  const wCmdZ = clamp(1.8 * errBody.z, -wmax, wmax);
  const ey = wCmdY - omega.y;
  const ez = wCmdZ - omega.z;
  const qn =
    (phase === "glide" || phase === "landing") && qDyn < 110_000
      ? Math.max(1, 1 + qDyn / 90_000)
      : Math.max(1, 1 + qDyn / 2_500);
  const finLim = (FIN_MAX_DEFLECT_RAD / qn) * g(gains, 8);
  const finEnable = qDyn < 4_000 ? 0 : 1;
  const finP = clamp((ey * 2.2) / qn, -finLim, finLim) * finEnable;
  const finY = clamp((ez * 2.2) / qn, -finLim, finLim) * finEnable;
  const finR = clamp((-omega.x * 1.4) / qn, -finLim, finLim) * finEnable;

  let gmax = 0;
  if ((phase === "landing" || phase === "glide") && qDyn < 20_000) {
    gmax =
      phase === "landing" && qDyn < 8_000 ? GIMBAL_MAX_RAD * 0.22 : GIMBAL_MAX_RAD;
  }
  gmax *= g(gains, 7);
  const gimY = clamp(-ey * 0.4, -gmax, gmax);
  const gimZ = clamp(-ez * 0.4, -gmax, gmax);
  return { gimY, gimZ, finP, finY, finR, errBody };
}

export function makeNav(
  p: Vec3,
  v: Vec3,
  bodyX: Vec3,
  q: number,
  aoa: number,
  mach: number,
  fuel: number,
  mass: number,
): Nav {
  const engine = p.add(bodyX.scale(-STAGE_LENGTH_M * 0.5));
  const rangeH = Math.hypot(p.x, p.y);
  const tilt = angleFromUp(bodyX);
  return {
    alt: p.z,
    engineAlt: engine.z,
    speed: v.len(),
    v,
    p,
    rangeH,
    tilt,
    q,
    aoa,
    mach,
    fuel,
    mass,
    bodyX,
  };
}

export function angleFromUp(bodyX: Vec3) {
  const d = clamp(bodyX.normalized().dot(new Vec3(0, 0, 1)), -1, 1);
  return Math.acos(d);
}

export function success(nav: Nav, intact: boolean) {
  const vh = Math.hypot(nav.v.x, nav.v.y);
  return (
    intact &&
    nav.engineAlt < SUCCESS_ENGINE_ALT_M &&
    nav.speed < SUCCESS_SPEED_MPS &&
    vh < SUCCESS_HVEL_MPS &&
    nav.rangeH < SUCCESS_PAD_OFFSET_M &&
    nav.tilt < SUCCESS_TILT_RAD
  );
}

/** Leg tips, not the engine bell — ~7 m Falcon 9 landing gear. */
export function groundHit(nav: Nav) {
  return nav.engineAlt < 6.5;
}

export function impactDestroy(nav: Nav) {
  return nav.speed > 20 || nav.tilt > (20 * Math.PI) / 180;
}

export function fuelInfeasible(nav: Nav, phase: Phase) {
  if (phase === "exo" || phase === "entry") return false;
  if (nav.fuel < 40 && nav.engineAlt > 40 && nav.speed > 25) return true;
  return false;
}
