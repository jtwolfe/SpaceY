import { lookup } from "./atmosphere";
import {
  DT,
  DRY_MASS_KG,
  FIN_SLEW_RAD_S,
  G0,
  GIMBAL_SLEW_RAD_S,
  IMPACT_SPEED_MPS,
  IMPACT_TILT_RAD,
  inertiaDiag,
  REF_AREA_M2,
  wetMass,
} from "./constants";
import {
  attitudeCommand,
  classifyPhase,
  emptyControls,
  fuelInfeasible,
  groundHit,
  makeNav,
  nominalControls,
  success,
  type Controls,
  type Gains,
  type Nav,
  type Phase,
  type TermReason,
  NOMINAL_GAINS,
} from "./guidance";
import { applyResidual, keepSinking, mlpForward, observe, N_HIDDEN, N_HIDDEN_LAYERS, N_OUT } from "./policy";
import { Quat, slew, Vec3 } from "./math";
import { spawnAt, type Spawn } from "./scenario";
import {
  aero,
  checkDestruction,
  EngineGate,
  propulsion,
  rcsMoment,
  type DestroyReason,
} from "./vehicle";

export type Pilot = "autopilot" | "manual" | "student";

export type ManualCmd = {
  pitch: number;
  yaw: number;
  roll: number;
  throttle: number;
  engines: number;
  fire: boolean;
};

export class Sim {
  t = 0;
  p = new Vec3();
  v = new Vec3();
  q = new Quat();
  omega = new Vec3();
  fuel = 0;
  timeout = 45;
  energy = 0;
  windScale = 0.2;
  windDir = 0;
  destroyEnabled = true;
  pilot: Pilot = "autopilot";
  gains: Gains = [...NOMINAL_GAINS];
  landingLatched = false;
  intact = true;
  destroyReason: DestroyReason = "none";
  term: TermReason = "none";
  phase: Phase = "landing";
  nav!: Nav;
  engine = new EngineGate();
  lastQ = 0;
  lastMach = 0;
  lastAoa = 0;
  lastHeat = 0;
  lastCd = 0;
  lastThrottle = 0;
  lastThrust = 0;
  lastN = 0;
  lastFins: [number, number, number] = [0, 0, 0];
  lastGimbal: [number, number] = [0, 0];
  lastAccelG = 1;
  lastDensity = 1.225;
  held = emptyControls();
  weights: number[] | null = null;
  lastY: number[] = Array.from({ length: N_OUT }, () => 0);
  lastHidden: number[] = Array.from({ length: N_HIDDEN * N_HIDDEN_LAYERS }, () => 0);
  manual: ManualCmd = {
    pitch: 0,
    yaw: 0,
    roll: 0,
    throttle: 0,
    engines: 1,
    fire: false,
  };
  seed = 1;
  minRange = 1e9;
  climbT = 0;

  static fromSpawn(spawn: Spawn, opts?: { destroy?: boolean; pilot?: Pilot; gains?: Gains; seed?: number; weights?: number[] }) {
    const s = new Sim();
    s.p = spawn.p.clone();
    s.v = spawn.v.clone();
    s.q = spawn.q.clone();
    s.omega = spawn.omega.clone();
    s.fuel = spawn.fuel;
    s.timeout = spawn.timeout;
    s.energy = spawn.energy;
    s.windScale = spawn.wind;
    s.destroyEnabled = opts?.destroy ?? spawn.energy > 0.3;
    s.pilot = opts?.pilot ?? "autopilot";
    s.gains = opts?.gains ? [...opts.gains] : [...NOMINAL_GAINS];
    s.weights = opts?.weights ? [...opts.weights] : null;
    s.seed = opts?.seed ?? 1;
    s.windDir = (s.seed % 360) * (Math.PI / 180);
    s.minRange = Math.hypot(s.p.x, s.p.y);
    s.climbT = 0;
    s.refreshNav();
    return s;
  }

  static start(energy: number, seed: number, opts?: { destroy?: boolean; pilot?: Pilot; gains?: Gains; weights?: number[] }) {
    return Sim.fromSpawn(spawnAt(energy, seed), { ...opts, seed });
  }

  terminated() {
    return this.term !== "none";
  }

  bodyX() {
    return this.q.rotate(new Vec3(1, 0, 0));
  }
  bodyY() {
    return this.q.rotate(new Vec3(0, 1, 0));
  }
  bodyZ() {
    return this.q.rotate(new Vec3(0, 0, 1));
  }

  refreshNav() {
    const mass = Math.max(DRY_MASS_KG, wetMass(this.fuel));
    this.nav = makeNav(
      this.p,
      this.v,
      this.bodyX(),
      this.lastQ,
      this.lastAoa,
      this.lastMach,
      this.fuel,
      mass,
    );
  }

  step(dt = DT) {
    if (this.terminated()) return;
    dt = Math.min(0.05, Math.max(0.001, dt));

    const air = lookup(this.p.z);
    this.lastDensity = air.density;
    const wSpeed = 14 * this.windScale * Math.min(1, this.p.z / 800);
    const vWind = new Vec3(Math.cos(this.windDir) * wSpeed, Math.sin(this.windDir) * wSpeed, 0);
    const vRelBody = this.q.conjugate().rotate(this.v.sub(vWind));

    this.refreshNav();
    if (!this.landingLatched && this.nav.rangeH < 8_000) {
      const vDown = Math.max(0, -this.nav.v.z);
      if (this.nav.engineAlt < 2_400 && vDown > 8) this.landingLatched = true;
      if (this.nav.engineAlt < 400) this.landingLatched = true;
    }
    this.phase = classifyPhase(this.nav, this.landingLatched);

    const bx = this.bodyX();
    const by = this.bodyY();
    const bz = this.bodyZ();

    let u: Controls;
    let desiredX: Vec3;
    if (this.pilot === "manual") {
      ({ u, desiredX } = this.manualControls(bx));
    } else {
      const nom = nominalControls(this.nav, this.phase, this.engine.on, this.gains);
      u = nom.u;
      desiredX = nom.desiredX;
      const att = attitudeCommand(bx, by, bz, this.omega, desiredX, this.phase, this.lastQ, this.gains);
      u.gimbalY = att.gimY;
      u.gimbalZ = att.gimZ;
      u.finPitch = att.finP;
      u.finYaw = att.finY;
      u.finRoll = att.finR;
      if (this.weights && this.weights.length) {
        const o = mlpForward(this.weights, observe(this.nav, by, this.omega));
        this.lastY = Array.from(o.y);
        this.lastHidden = [...Array.from(o.h1), ...Array.from(o.h2)];
        applyResidual(u, o.y);
        keepSinking(u, this.nav);
      }
    }

    const gated = this.engine.apply(this.t, u.throttle, u.nEngines);
    u.throttle = gated.throttle;
    u.nEngines = gated.n;
    u.gimbalY = slew(this.lastGimbal[0], u.gimbalY, GIMBAL_SLEW_RAD_S, dt);
    u.gimbalZ = slew(this.lastGimbal[1], u.gimbalZ, GIMBAL_SLEW_RAD_S, dt);
    u.finPitch = slew(this.lastFins[0], u.finPitch, FIN_SLEW_RAD_S, dt);
    u.finYaw = slew(this.lastFins[1], u.finYaw, FIN_SLEW_RAD_S, dt);
    u.finRoll = slew(this.lastFins[2], u.finRoll, FIN_SLEW_RAD_S, dt);

    const a = aero(vRelBody, air, u.finPitch, u.finYaw, u.finRoll);
    const burn = propulsion(u.throttle, u.gimbalY, u.gimbalZ, u.nEngines, air.pressurePa, this.fuel);

    this.lastQ = a.q;
    this.lastMach = a.mach;
    this.lastAoa = a.aoa;
    this.lastHeat = a.heat;
    this.lastCd = a.cd;
    this.lastThrottle = u.throttle;
    this.lastThrust = burn.thrust;
    this.lastN = u.nEngines;
    this.lastFins = [u.finPitch, u.finYaw, u.finRoll];
    this.lastGimbal = [u.gimbalY, u.gimbalZ];

    const mass = Math.max(DRY_MASS_KG, wetMass(this.fuel));
    const fBody = a.forceBody.add(burn.forceBody);
    const damp = a.q * REF_AREA_M2 * 28;
    const w0 = this.omega;
    const err = bx.cross(desiredX.normalized());
    const errBody = new Vec3(err.dot(bx), err.dot(by), err.dot(bz));
    const i = inertiaDiag(mass);
    const mRcs =
      this.pilot === "manual"
        ? rcsMoment(this.omega, errBody, i, a.q)
        : rcsMoment(this.omega, errBody, i, a.q);
    const mBody = a.momentBody
      .add(burn.momentBody)
      .add(mRcs)
      .add(new Vec3(-damp * 1.1 * w0.x, -damp * w0.y, -damp * w0.z));
    const aBody = fBody.scale(1 / mass);
    const gLocal = new Vec3(0, 0, -G0 * (EARTH_G(this.p.z)));
    const aWorld = gLocal.add(this.q.rotate(aBody));
    this.lastAccelG = aWorld.len() / G0;

    let wdot = new Vec3(
      (mBody.x - (w0.y * (i.z * w0.z) - w0.z * (i.y * w0.y))) / Math.max(1, i.x),
      (mBody.y - (w0.z * (i.x * w0.x) - w0.x * (i.z * w0.z))) / Math.max(1, i.y),
      (mBody.z - (w0.x * (i.y * w0.y) - w0.y * (i.x * w0.x))) / Math.max(1, i.z),
    );
    const wdotMax = a.q > 8_000 ? 1.6 : 2.4;
    wdot = wdot.clampLen(wdotMax);

    if (this.pilot === "manual") {
      const m = this.manual;
      const wWorld = new Vec3(-m.pitch * 0.95, -m.yaw * 0.95, m.roll * 0.8);
      const wCmd = this.q.conjugate().rotate(wWorld);
      wdot = wdot.add(wCmd.sub(this.omega).scale(10)).clampLen(4);
    }

    this.v = this.v.add(aWorld.scale(dt));
    this.p = this.p.add(this.v.scale(dt));
    this.omega = this.omega.add(wdot.scale(dt));
    this.q = this.q.integrate(this.omega, dt);
    this.fuel = Math.max(0, this.fuel - burn.mdot * dt);
    this.t += dt;

    this.refreshNav();
    this.minRange = Math.min(this.minRange, this.nav.rangeH);
    if (this.nav.engineAlt < 3_000 && this.v.z > 0.8) this.climbT += dt;

    const dest = checkDestruction(this.lastQ, this.lastAoa, this.lastAccelG, this.omega.len(), this.destroyEnabled);
    if (dest !== "none") {
      this.intact = false;
      this.destroyReason = dest;
      this.term = "destroyed";
      return;
    }

    if (groundHit(this.nav)) {
      if (success(this.nav, this.intact)) {
        this.term = "landed";
      } else if (
        this.destroyEnabled &&
        (this.nav.speed > IMPACT_SPEED_MPS || this.nav.tilt > IMPACT_TILT_RAD)
      ) {
        this.intact = false;
        this.destroyReason = "impact";
        this.term = "destroyed";
      } else {
        this.term = "miss";
      }
      this.v = new Vec3();
      this.omega = new Vec3();
      return;
    }

    if (fuelInfeasible(this.nav, this.phase)) {
      this.term = "fuel";
      return;
    }
    if (this.t > this.timeout) this.term = "timeout";
  }

  stepFor(seconds: number) {
    let acc = 0;
    while (acc < seconds && !this.terminated()) {
      const dt = Math.min(DT, seconds - acc);
      this.step(dt);
      acc += dt;
    }
  }

  private manualControls(bx: Vec3): { u: Controls; desiredX: Vec3 } {
    const u = emptyControls();
    const m = this.manual;
    if (m.fire && m.throttle > 0.02) {
      u.throttle = m.throttle;
      u.nEngines = m.engines >= 2 ? 3 : 1;
    }
    const up = new Vec3(0, 0, 1);
    const east = new Vec3(1, 0, 0);
    const north = new Vec3(0, 1, 0);
    // A = +yaw → tilt top toward -east from a south-looking chase cam (nose left).
    const tilt = 0.5;
    let desired = up
      .add(east.scale(-m.yaw * tilt))
      .add(north.scale(m.pitch * tilt))
      .normalized();
    if (this.nav.engineAlt > 400 && this.v.len() > 40) {
      desired = this.v.neg().normalized().lerp(desired, 0.35).normalized();
    }
    const att = attitudeCommand(
      bx,
      this.bodyY(),
      this.bodyZ(),
      this.omega,
      desired,
      this.phase,
      this.lastQ,
      this.gains,
    );
    u.gimbalY = att.gimY;
    u.gimbalZ = att.gimZ;
    u.finPitch = att.finP + m.pitch * 0.15;
    u.finYaw = att.finY + m.yaw * 0.15;
    u.finRoll = att.finR + m.roll * 0.2;
    return { u, desiredX: desired };
  }
}

function EARTH_G(alt: number) {
  const re = 6_378_137;
  const r = re / (re + Math.max(0, alt));
  return r * r;
}

export function snapshot(sim: Sim) {
  return {
    t: sim.t,
    alt: sim.nav.alt,
    engineAlt: sim.nav.engineAlt,
    speed: sim.nav.speed,
    range: sim.nav.rangeH,
    tiltDeg: (sim.nav.tilt * 180) / Math.PI,
    fuel: sim.fuel,
    phase: sim.phase,
    term: sim.term,
    qkpa: sim.lastQ / 1000,
    mach: sim.lastMach,
    aoaDeg: (sim.lastAoa * 180) / Math.PI,
    throttle: sim.lastThrottle,
    nEngines: sim.lastN,
    thrust: sim.lastThrust,
    g: sim.lastAccelG,
    intact: sim.intact,
    destroy: sim.destroyReason,
    p: sim.p,
    v: sim.v,
    bodyX: sim.bodyX(),
    energy: sim.energy,
  };
}

export type Snapshot = ReturnType<typeof snapshot>;


