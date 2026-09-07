import {
  FIN_MAX_DEFLECT_RAD,
  G0,
  GIMBAL_MAX_RAD,
  MERLIN_THRUST_SL_N,
  N_ENGINES_ENTRY,
  N_ENGINES_LANDING,
  REF_AREA_M2,
  STAGE_LENGTH_M,
  SUCCESS_ENGINE_ALT_M,
  SUCCESS_HVEL_MPS,
  SUCCESS_PAD_OFFSET_M,
  SUCCESS_SPEED_MPS,
  SUCCESS_TILT_RAD,
  SUICIDE_FUEL_KG,
  THROTTLE_MIN,
  GEAR_ENGINE_ALT_M,
} from "./constants";
import { lookup } from "./atmosphere";
import { clamp, lerp, saturate, Vec3 } from "./math";
import { finQEnable } from "./vehicle";

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
  wind: Vec3;
};

export type Gains = number[];
/** ignite, kp, kd, maxTilt, entry, dive, att, tvc, fin, three, thr, yaw */
export const N_GAINS = 12;
export const NOMINAL_GAINS: Gains = Array.from({ length: N_GAINS }, () => 1);

export function classifyPhase(nav: Nav, landingLatched: boolean, energy = 0): Phase {
  // RTLS: never treat a high-altitude pad overflight as a suicide window.
  // Pad / 2 km / Glide keep the original ladder below.
  if (energy >= 0.85) {
    if (landingLatched) return "landing";
    if (rtlsSuicideWindow(nav)) return "landing";
    if (nav.alt > 78_000 && nav.q < 80 && nav.speed < 400) return "exo";
    if (nav.speed > 900 && nav.alt > 35_000) return "entry";
    if (nav.alt > 2_600) return "glide";
    return "landing";
  }
  if (landingLatched) return "landing";
  if (shouldStartLanding(nav)) return "landing";
  if (nav.alt > 78_000 && nav.q < 80 && nav.speed < 400) return "exo";
  const pred = predictedLandingRange(nav);
  const long = pred > nav.rangeH + 8_000 && nav.speed > 450 && nav.alt > 12_000;
  if (long) return "entry";
  if (nav.rangeH < 5_000 && nav.speed < 520 && !long) {
    if (nav.alt > 8_000) return "glide";
    if (hotUnpoweredCoast(nav)) return "glide";
    return "landing";
  }
  // Floor 600 m/s so a ~400 m/s glide spawn is not "entry" (weak fins / no energy bleed).
  if (nav.alt > 12_000 && nav.speed > vRef(nav.alt) + 40 && nav.speed > 600) return "entry";
  if (nav.speed > 1_550 && nav.alt < 105_000) return "entry";
  if (nav.alt > 8_000) return "glide";
  if (hotUnpoweredCoast(nav)) return "glide";
  return "landing";
}

/** Still fast and high-q: stay on unpowered glide aim, not the near-vertical landing lean. */
function hotUnpoweredCoast(nav: Nav) {
  return nav.speed > 210 && nav.q > 8_000 && nav.alt > 2_600;
}

/** RTLS: light only when 1 Merlin + some drag still needs a real suicide, not a 5 km hover. */
export function rtlsSuicideWindow(nav: Nav) {
  const pz = nav.engineAlt;
  const vDown = Math.max(0, -nav.v.z);
  // Deck must light. A 1.6 km force-light with small vDown is a 40% Merlin hover
  // (min throttle still climbs an empty booster) that burns out 1 km over the disk.
  if (pz < 280) return true;
  if (pz < 1_600 && vDown > 22) return true;
  const sl = suicideLightAlt(nav);
  if (sl.use3 && pz > 2_200) return false;
  const aEng = clamp(MERLIN_THRUST_SL_N / nav.mass - G0, 4, 40);
  const aDrag = (nav.q * REF_AREA_M2 * 0.4) / Math.max(1, nav.mass);
  const s1 = Math.max(0, vDown * vDown - 36) / (2 * (aEng + aDrag)) + 10;
  return pz < s1 * 1.08 && pz < 4_200;
}

export function vRef(alt: number) {
  if (alt > 70_000) return 1_800;
  if (alt > 45_000) return 550 + (1_000 * (alt - 45_000)) / 25_000;
  if (alt > 20_000) return 280 + (270 * (alt - 20_000)) / 25_000;
  if (alt > 8_000) return 120 + (160 * (alt - 8_000)) / 12_000;
  return 50 + (70 * alt) / 8_000;
}

/** Altitude where a 1-engine suicide should light. Three-wide is a pulse, not the clock. */
export function suicideLightAlt(nav: Nav, gains: Gains = NOMINAL_GAINS): { sLight: number; use3: boolean } {
  const pz = Math.max(0.5, nav.engineAlt);
  const vDown = Math.max(0, -nav.v.z);
  const vLand = 6.0;
  const hLand = 4.0;
  const a1 = clamp(MERLIN_THRUST_SL_N / nav.mass - G0, 4, 40);
  const margin = vDown > 180 && nav.engineAlt > 1_500 ? 0.82 : 0.96;
  const s1 = Math.max(0, vDown * vDown - vLand * vLand) / (2 * a1 * margin) + hLand;
  const cluster = landingEngineCluster(nav);
  const sLight = s1 * g(gains, 0);
  return { sLight, use3: cluster.use3 && pz > 110 };
}

/** 1-engine landing default. 3-wide only when one Merlin at 100% cannot meet needA
 *  AND the 3-wide throttle would stay above the floor. Floor clamp → stay on 1 @ 100%. */
export function landingEngineCluster(nav: Nav): { n: number; throttle: number; use3: boolean } {
  const pz = Math.max(0.5, nav.engineAlt);
  const vDown = Math.max(0, -nav.v.z);
  const hLand = 4.0;
  const vLand = 6.0;
  const needA = Math.max(0, (vDown * vDown - vLand * vLand) / (2 * Math.max(pz - hLand, 4)));
  const tiltFac = 1 / Math.max(0.72, Math.cos(Math.min(0.5, nav.tilt)));
  let thr1 = ((needA + G0) * nav.mass * tiltFac) / MERLIN_THRUST_SL_N;
  if (pz < 18 && vDown > 4) thr1 = Math.max(thr1, 1);
  if (thr1 <= 1) {
    return { n: N_ENGINES_LANDING, throttle: Math.max(THROTTLE_MIN, saturate(thr1)), use3: false };
  }
  const thr3 = ((needA + G0) * nav.mass * tiltFac) / (MERLIN_THRUST_SL_N * N_ENGINES_ENTRY);
  if (thr3 < THROTTLE_MIN + 0.05) {
    return { n: N_ENGINES_LANDING, throttle: 1, use3: false };
  }
  return {
    n: N_ENGINES_ENTRY,
    throttle: Math.max(THROTTLE_MIN, saturate(thr3)),
    use3: true,
  };
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

/** Drag-aware ground intercept. Used by unpowered glide steering, not entry-burn range. */
function predictedImpact(nav: Nav): { x: number; y: number; t: number } {
  let px = nav.p.x;
  let py = nav.p.y;
  let pz = nav.p.z;
  let vx = nav.v.x;
  let vy = nav.v.y;
  let vz = nav.v.z;
  const mass = Math.max(1, nav.mass);
  const sAoa = Math.sin(Math.min(1.2, Math.abs(nav.aoa)));
  const cdA = (0.95 + 1.55 * sAoa * sAoa) * REF_AREA_M2;
  let t = 0;
  for (let i = 0; i < 16; i++) {
    if (pz < 10) break;
    const vd = Math.max(10, -vz);
    const dt = clamp(pz / (vd * 5), 0.35, 3.8);
    const air = lookup(pz);
    const spd = Math.hypot(vx, vy, vz);
    const q = 0.5 * air.density * spd * spd;
    const aDrag = (cdA * q) / mass;
    const inv = spd > 1 ? aDrag / spd : 0;
    const ax = -vx * inv;
    const ay = -vy * inv;
    const az = -vz * inv - G0;
    vx += ax * dt;
    vy += ay * dt;
    vz += az * dt;
    px += vx * dt;
    py += vy * dt;
    pz += vz * dt;
    t += dt;
    if (pz <= 0) {
      const back = pz / Math.min(-1e-3, vz);
      px -= vx * back;
      py -= vy * back;
      t -= back;
      pz = 0;
      break;
    }
  }
  return { x: px, y: py, t };
}

/** High-altitude intercept. Glide steering keeps the 16-step `predictedImpact`. */
function predictedImpactRtls(nav: Nav): { x: number; y: number; t: number } {
  let px = nav.p.x;
  let py = nav.p.y;
  let pz = nav.p.z;
  let vx = nav.v.x;
  let vy = nav.v.y;
  let vz = nav.v.z;
  const mass = Math.max(1, nav.mass);
  const sAoa = Math.sin(Math.min(1.2, Math.abs(nav.aoa)));
  const cdA = (0.95 + 1.55 * sAoa * sAoa) * REF_AREA_M2;
  let t = 0;
  for (let i = 0; i < 48; i++) {
    if (pz < 10) break;
    const vd = Math.max(10, -vz);
    const dt = clamp(pz / (vd * 4), 0.4, 8);
    const air = lookup(pz);
    const spd = Math.hypot(vx, vy, vz);
    const q = 0.5 * air.density * spd * spd;
    const aDrag = (cdA * q) / mass;
    const inv = spd > 1 ? aDrag / spd : 0;
    vx += -vx * inv * dt;
    vy += -vy * inv * dt;
    vz += (-vz * inv - G0) * dt;
    px += vx * dt;
    py += vy * dt;
    pz += vz * dt;
    t += dt;
    if (pz <= 0) {
      const back = pz / Math.min(-1e-3, vz);
      px -= vx * back;
      py -= vy * back;
      t -= back;
      pz = 0;
      break;
    }
  }
  return { x: px, y: py, t };
}

/** RTLS-only: 3-Merlin entry until drag-aware intercept is near the pad. */
function rtlsEntryCommand(nav: Nav): { want: boolean; still: boolean; thr: number } {
  // 0.95×suicide left hot seeds hanging 20 km past the disk — they hit reserve
  // with extra still > 2 km. Hero cuts on extra, not reserve, so this is free.
  const reserve = SUICIDE_FUEL_KG * 0.72;
  const fuelOk = nav.fuel > reserve;
  const imp = predictedImpactRtls(nav);
  const predRange = Math.hypot(imp.x, imp.y);
  const past = nav.p.x * imp.x + nav.p.y * imp.y < 0;
  const extra = past ? predRange : -predRange;
  const hypersonic = nav.speed > 1_450;
  const qHot = nav.q > 28_000 && nav.speed > 500;
  const long = extra > 3_500;
  const high = nav.alt > 38_000 && nav.alt < 95_000;
  // 3-wide through a loft (vz→0) pitches the stack past retro and somersaults.
  const falling = nav.v.z < -70;
  const want = fuelOk && high && falling && (hypersonic || qHot || long);
  const still = fuelOk && high && falling && (hypersonic || qHot || extra > 2_000);
  let thr = 1;
  if (!hypersonic && extra < 14_000) thr = clamp(0.55 + extra / 22_000, 0.55, 1);
  if (qHot && nav.q > 60_000) thr = 1;
  return { want, still, thr };
}

function g(gains: Gains, i: number) {
  return gains[i] ?? 1;
}

export type EntryBurn = "idle" | "on" | "done";

export type BurnLatch = {
  energy: number;
  entry: EntryBurn;
  landingDone: boolean;
};

export function nominalControls(
  nav: Nav,
  phase: Phase,
  engineOn: boolean,
  gains: Gains,
  latch: BurnLatch,
): { u: Controls; desiredX: Vec3 } {
  let desiredX = new Vec3(0, 0, 1);
  const u = emptyControls();
  const rtls = latch.energy >= 0.85;

  // RTLS entry is independent of phase so a 450 m/s glide flicker cannot one-shot it.
  if (rtls && latch.entry !== "done" && phase !== "landing") {
    const cmd = rtlsEntryCommand(nav);
    if (latch.entry === "on" || cmd.want) {
      if (cmd.still) {
        latch.entry = "on";
        desiredX = aimRetro(nav, gains);
        u.nEngines = N_ENGINES_ENTRY;
        u.throttle = cmd.thr;
        return { u, desiredX };
      }
      latch.entry = "done";
    }
  }

  if (phase === "exo" || phase === "entry") {
    if (rtls) {
      desiredX = unpoweredGlideAim(nav, gains, rtls);
      u.nEngines = 0;
    } else if (nav.v.len() > 10) {
      desiredX = aimRetro(nav, gains);
      u.nEngines = 0;
    } else {
      desiredX = unpoweredGlideAim(nav, gains, rtls);
      u.nEngines = 0;
    }
  } else if (phase === "glide") {
    if (!rtls && latch.entry === "on") latch.entry = "done";
    desiredX = unpoweredGlideAim(nav, gains, rtls);
    u.nEngines = 0;
  } else {
    if (!rtls && latch.entry === "on") latch.entry = "done";
    hoverSlam(nav, u, (x) => {
      desiredX = x;
    }, engineOn, gains, latch.landingDone, rtls);
    if (!engineOn && hotUnpoweredCoast(nav) && u.nEngines <= 0) {
      desiredX = unpoweredGlideAim(nav, gains, rtls);
    }
  }

  return { u, desiredX };
}

function landingTgo(nav: Nav) {
  const pz = Math.max(0.5, nav.engineAlt);
  const vz = nav.v.z;
  if (pz <= 0) return 0;
  if (vz >= 0) {
    const tUp = vz / G0;
    const zPeak = pz + vz * tUp - 0.5 * G0 * tUp * tUp;
    return tUp + Math.sqrt(Math.max(0, (2 * zPeak) / G0));
  }
  const vDown = -vz;
  return (Math.sqrt(Math.max(0, vDown * vDown + 2 * G0 * pz)) - vDown) / G0;
}

/** Unpowered high-altitude aim: tail-first, pitch toward zenith for drag, bank for miss. */
export function unpoweredGlideAim(nav: Nav, gains: Gains = NOMINAL_GAINS, rtls = false): Vec3 {
  const spd = nav.speed;
  if (spd < 12) return unpoweredLandingAim(nav);
  const vh = Math.hypot(nav.v.x, nav.v.y);
  const vDown = Math.max(0, -nav.v.z);
  const steep = vDown / Math.max(spd, 1) > 0.88;
  if (nav.alt < 3_200 && spd < 190 && nav.rangeH < 2_800) return unpoweredLandingAim(nav);
  if (nav.alt < 3_800 && steep && nav.rangeH < 2_200 && spd < 220) return unpoweredLandingAim(nav);
  if (rtls && nav.alt < 8_500 && vh < 90 && nav.rangeH > 40 && nav.rangeH < 2_200 && spd > 90) {
    return unpoweredLandingAim(nav);
  }

  const vhat = nav.v.scale(1 / spd);
  const retro = vhat.neg();
  const up = new Vec3(0, 0, 1);
  const imp = predictedImpact(nav);
  const range = Math.max(1, nav.rangeH);
  const predRange = Math.hypot(imp.x, imp.y);
  const predPast = nav.p.x * imp.x + nav.p.y * imp.y < 0;
  const rHat = new Vec3(nav.p.x / range, nav.p.y / range, 0);
  const predRad = imp.x * rHat.x + imp.y * rHat.y;
  const extra = predPast ? predRange + Math.max(0, -predRad) : -predRad;

  const q = nav.q;
  let aoaMax = 1.15;
  if (rtls && q > 12_000) {
    // Stay under AoA destroy (42°) and q-alpha 2800; glide rung keeps the tighter schedule.
    const qk = q / 1000;
    const qAlphaMax = ((2_400 / Math.max(12, qk)) * Math.PI) / 180;
    aoaMax = Math.min(0.62, qAlphaMax);
    if (q > 90_000) aoaMax = Math.min(aoaMax, 0.32);
  } else if (q > 28_000) aoaMax = 0.36;
  else if (q > 16_000) aoaMax = lerp(0.48, 0.36, (q - 16_000) / 12_000);
  else if (q > 11_500) aoaMax = lerp(0.55, 0.48, (q - 11_500) / 4_500);
  else if (q > 8_000) aoaMax = lerp(0.95, 0.55, (q - 8_000) / 3_500);
  else if (q > 5_000) aoaMax = lerp(1.15, 0.95, (q - 5_000) / 3_000);

  const energy = ((extra + Math.max(0, predRange - 160)) / 6_500) * g(gains, 5);
  let aoaCmd: number;
  if (extra > 150 || predPast || vh > 70) {
    aoaCmd = aoaMax;
  } else if (extra < -200) {
    aoaCmd = clamp(0.06 + Math.max(0, energy) * 0.12, 0.04, aoaMax * 0.4);
  } else {
    aoaCmd = clamp(0.2 + Math.abs(energy) * 0.25, 0.12, aoaMax);
  }
  if (rtls) {
    // Vacuum AoA is a flip, not drag. At 3-Merlin cutoff the command used to snap
    // 66° toward zenith, leftover TVC rate became a somersault, and body lift
    // sprayed half the population off the corridor. Hold tail-first until q
    // can actually bleed energy; full AoA by ~7 kPa.
    aoaCmd *= saturate((q - 80) / 7_000);
  }

  const pitchDir = up.sub(retro.scale(retro.dot(up)));
  if (pitchDir.len() < 0.06) return retro.lerp(up, 0.2).normalized();
  let steer = pitchDir.normalized();
  const missH = new Vec3(imp.x, imp.y, 0);
  const recede = nav.p.x * nav.v.x + nav.p.y * nav.v.y > 0 && range > 60;
  if (missH.len() > 80 || recede) {
    const lat = recede ? rHat.neg() : missH.neg();
    const latP = lat.sub(retro.scale(lat.dot(retro)));
    if (latP.len() > 0.05) {
      const bank = clamp((recede ? range : missH.len()) / (rtls ? 5_500 : 8_000), 0.04, rtls ? 0.28 : 0.18);
      steer = steer.lerp(latP.normalized(), bank);
      const sn = steer.len();
      if (sn > 0.05) steer = steer.scale(1 / sn);
    }
  }

  const desired = retro.scale(Math.cos(aoaCmd)).add(steer.scale(Math.sin(aoaCmd)));
  const n = desired.len();
  return n < 0.2 ? retro : desired.scale(1 / n);
}

/** Unpowered landing aim: near vertical / into the wind, engines toward the pad for body lift. */
export function unpoweredLandingAim(nav: Nav): Vec3 {
  const up = new Vec3(0, 0, 1);
  const range = nav.rangeH;
  const wind = nav.wind ?? Vec3.ZERO;
  const vAir = nav.v.sub(wind);
  const tG = Math.max(0.2, landingTgo(nav));
  const predX = nav.p.x + nav.v.x * tG + 0.35 * wind.x * tG;
  const predY = nav.p.y + nav.v.y * tG + 0.35 * wind.y * tG;
  const predRange = Math.hypot(predX, predY);

  let desired = up;
  const airSpd = vAir.len();
  if (airSpd > 18) {
    const retro = vAir.neg().normalized();
    desired = up.lerp(retro, 0.18).normalized();
    if (desired.len() < 0.2) desired = up;
  }

  if (range < 22 && predRange < 28) return desired.lerp(up, 0.55).normalized();
  if (predRange < 18) return desired;

  const away = new Vec3(nav.p.x, nav.p.y, 0);
  if (away.len() < 1) return desired;
  const pull = clamp(Math.max(predRange, range) / 140, 0.14, 0.52);
  const lift = up.scale(Math.cos(pull)).add(away.normalized().scale(Math.sin(pull))).normalized();
  const mixed = desired.lerp(lift, 0.88);
  return mixed.len() < 0.2 ? lift : mixed.normalized();
}

function hoverSlam(
  nav: Nav,
  u: Controls,
  setDesired: (x: Vec3) => void,
  engineOn: boolean,
  gains: Gains,
  landingDone: boolean,
  rtls = false,
) {
  if (landingDone) {
    u.throttle = 0;
    u.nEngines = 0;
    setDesired(new Vec3(0, 0, 1));
    return;
  }

  const pz = Math.max(0.5, nav.engineAlt);
  const vDown = Math.max(0, -nav.v.z);
  const climbing = nav.v.z > 3;
  const hLand = 4.0;
  const { sLight } = suicideLightAlt(nav, gains);

  const vh = Math.hypot(nav.v.x, nav.v.y);
  const range = nav.rangeH;
  const overPad = range < 22 && vh < 5.5;
  const translating = !overPad && range > 22;

  if (!engineOn && pz > sLight + 8 && pz > hLand + 10) {
    u.throttle = 0;
    u.nEngines = 0;
  } else {
    const cluster = landingEngineCluster(nav);
    u.nEngines = cluster.n;
    u.throttle = cluster.throttle;
  }

    if (engineOn && pz > GEAR_ENGINE_ALT_M && (u.nEngines <= 0 || u.throttle <= 0.02)) {
    u.nEngines = N_ENGINES_LANDING;
    u.throttle = THROTTLE_MIN;
  }

  if (climbing && engineOn && pz > GEAR_ENGINE_ALT_M) {
    u.nEngines = N_ENGINES_LANDING;
    u.throttle = THROTTLE_MIN;
  }

  if (engineOn && overPad && pz < 70 && vDown > 6 && !climbing) {
    u.nEngines = Math.max(u.nEngines, N_ENGINES_LANDING);
    u.throttle = Math.max(u.throttle, 0.85);
  }

  if (rtls && overPad && pz < 16 && (climbing || vDown < 9)) {
    u.throttle = 0;
    u.nEngines = 0;
  }

  const thrusting = u.nEngines > 0 && u.throttle > 0.02;
  if (!thrusting) {
    setDesired(unpoweredLandingAim(nav));
    return;
  }

  let azCmd = Math.max(2, (u.throttle * MERLIN_THRUST_SL_N * u.nEngines) / nav.mass);
  const ahMax = Math.max(2, azCmd * Math.sin(0.35));
  const sBrake = (vh * vh) / (2 * ahMax) + 20;
  const closing = range < 1 || nav.p.x * nav.v.x + nav.p.y * nav.v.y < 0;
  const tG = Math.max(0.2, landingTgo(nav));
  const predX = nav.p.x + nav.v.x * tG;
  const predY = nav.p.y + nav.v.y * tG;
  const predPast = nav.p.x * predX + nav.p.y * predY < 0;
  const cross = nav.p.x * nav.v.y - nav.p.y * nav.v.x;
  const b = Math.abs(cross) / Math.max(vh, 1e-3);
  const willCross = vh > 2 && closing && b < 22 && range > b;
  const trueOvershoot = translating && willCross && range < sBrake && (predPast || b < 18);

  const aUp = Math.max(4, azCmd * Math.cos(0.25) - G0);
  const tBurn = Math.max(0.5, vDown / aUp);
  const dvH = ahMax * tBurn;
  const needH = vh + range / tBurn;
  const killFrac = trueOvershoot ? 1 : needH < 1e-3 ? 1 : clamp(dvH / needH, 0.2, 1);

  let kp = (pz < 80 ? 0.32 : 0.18) * g(gains, 1);
  let kd = (pz < 80 ? 2.6 : 2.0) * g(gains, 2);
  const predRange = Math.hypot(predX, predY);
  const skid = predRange < 45 && vh > 7;
  let ax: number;
  let ay: number;
  if (trueOvershoot || skid) {
    kp *= trueOvershoot ? 0.35 : 0.22;
    kd *= trueOvershoot ? 1.4 : 1.7;
    ax = -kp * nav.p.x - kd * nav.v.x;
    ay = -kp * nav.p.y - kd * nav.v.y;
  } else {
    ax = -kp * predX * killFrac;
    ay = -kp * predY * killFrac;
  }

  const horiz = Math.hypot(ax, ay);
  let maxTilt = landingMaxTilt(pz, predRange, predRange > 22 || vh > 8) * g(gains, 3);
  if (!trueOvershoot && !skid) maxTilt *= killFrac;
  if (skid) maxTilt = Math.max(maxTilt, clamp(vh / 70, 0.08, 0.28));
  if (rtls && nav.q > 8_000) maxTilt = Math.min(maxTilt, 0.12);
  if (rtls && nav.q > 16_000) maxTilt = Math.min(maxTilt, 0.07);
  if (rtls && pz < 480) maxTilt = Math.min(maxTilt, 0.16);
  if (rtls && pz < 160) maxTilt = Math.min(maxTilt, 0.09);
  const enginesTowardPad =
    range > 1 && (nav.p.x * nav.bodyX.x + nav.p.y * nav.bodyX.y) / range > 0.08;
  if (enginesTowardPad && !trueOvershoot && !skid && pz < 1_200) maxTilt *= 0.5;

  let tilt = Math.min(maxTilt, horiz / Math.max(2, azCmd));

  if (translating && climbing) {
    const vert = azCmd * Math.cos(Math.max(0.05, tilt));
    if (vert > G0 * 0.92) {
      const maxAz = (G0 * 0.86) / Math.max(0.25, Math.cos(Math.max(tilt, 0.2)));
      u.throttle = clamp(
        (maxAz * nav.mass) / (MERLIN_THRUST_SL_N * Math.max(1, u.nEngines)),
        THROTTLE_MIN,
        u.throttle,
      );
      azCmd = (u.throttle * MERLIN_THRUST_SL_N * u.nEngines) / nav.mass;
    }
  }

  if (u.nEngines >= N_ENGINES_ENTRY && u.throttle <= THROTTLE_MIN + 0.03) {
    u.nEngines = N_ENGINES_LANDING;
    if (vDown > 40) u.throttle = 1;
    azCmd = (u.throttle * MERLIN_THRUST_SL_N * u.nEngines) / nav.mass;
  }

  tilt = Math.min(tilt, 0.55);

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

/** Powered leftover miss: small TVC, upright near the disk. Unpowered lean is engines-toward-pad, not this. */
function landingMaxTilt(pz: number, range: number, translating: boolean) {
  if (!translating || range < 22) {
    if (pz < 28) return 0.04;
    if (pz < 70) return 0.09;
    if (pz < 200) return 0.2;
    return 0.28;
  }
  const byRange = clamp(range / (pz > 1_200 ? 160 : 220), 0.08, pz > 1_200 ? 0.55 : 0.38);
  if (pz < 18) return Math.min(byRange, 0.08);
  if (pz < 36) return Math.min(byRange, 0.2);
  return byRange;
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
  else if (phase === "glide" && qDyn < 400) wmax = 0.14;
  else if (phase === "glide" && qDyn < 110_000) wmax = 0.42;
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
  const finEnable = finQEnable(qDyn);
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
  wind: Vec3 = new Vec3(),
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
    wind,
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
