import { DT } from "./constants";
import { createInput, readActions } from "./input";
import { snapshot, Sim, type Pilot } from "./sim";
import { SpaceyScene } from "./scene";
import { Trainer, type Brain, type GymSnap } from "./trainer";
import type { Mission } from "./scenario";
import { energyForMission, missionLabel, snapEnergy } from "./scenario";
import type { AppPilot, CamMode, HudSnap } from "./store";

export type RuntimeHooks = {
  getPilot: () => AppPilot;
  getMission: () => Mission;
  getCam: () => CamMode;
  getWarp: () => number;
  getPaused: () => boolean;
  getStarted: () => boolean;
  getSeed: () => number;
  getBrain: () => Brain;
  getBrainEpoch: () => number;
  onSnap: (s: HudSnap) => void;
  onBrain: (b: Brain) => void;
  onGym: (g: GymSnap) => void;
  onNote: (n: string) => void;
};

export function createRuntime(canvas: HTMLCanvasElement, hooks: RuntimeHooks) {
  const scene = new SpaceyScene(canvas);
  const input = createInput();
  const qaKeys = new Set<string>();
  const trainer = new Trainer();
  trainer.applyBrain(hooks.getBrain());

  let sim = bootSolo();
  let watchHold = 0;
  let watchAcc = 0;
  let watch = bootWatch();
  let acc = 0;
  let last = performance.now();
  let running = true;
  let lastMission = hooks.getMission();
  let lastPilot = hooks.getPilot();
  let lastSeed = hooks.getSeed();
  let lastEpoch = hooks.getBrainEpoch();
  let uiT = 0;
  let appliedWarp = 1;

  function energyNow() {
    if (hooks.getPilot() === "train") return snapEnergy(trainer.brain.energy);
    return energyForMission(hooks.getMission());
  }

  function bootSolo() {
    const pilot = hooks.getPilot();
    const simPilot: Pilot = pilot === "manual" ? "manual" : "autopilot";
    return Sim.start(energyNow(), hooks.getSeed(), {
      pilot: simPilot,
      destroy: energyNow() > 0.3,
    });
  }

  function bootWatch() {
    const spec = trainer.takeWatch();
    watchHold = 0;
    watchAcc = 0;
    return Sim.start(spec.energy, spec.seed, {
      destroy: spec.energy > 0.3,
      pilot: "student",
      weights: spec.weights,
    });
  }

  function recycleWatch(handoff = true) {
    watch = bootWatch();
    scene.resetTrail();
    if (handoff) scene.beginHandoff();
  }

  function applyNetReset() {
    trainer.applyBrain(hooks.getBrain());
    recycleWatch(false);
    scene.resetLook();
  }

  function restart() {
    if (hooks.getPilot() === "train") {
      recycleWatch(false);
      scene.resetLook();
    } else {
      sim = bootSolo();
      scene.resetTrail();
      scene.resetLook();
    }
  }

  function viewed() {
    return hooks.getPilot() === "train" ? watch : sim;
  }

  const probe = {
    getYaw: () => {
      const b = viewed().bodyX();
      return Math.atan2(-b.x, b.z);
    },
    getSpeed: () => viewed().v.len(),
    setKeys: (codes: string[]) => {
      qaKeys.clear();
      for (const c of codes) qaKeys.add(c);
    },
    look: (dyaw: number, dpitch: number, zoomMul = 1) => scene.look(dyaw, dpitch, zoomMul),
    getLook: () => scene.getLook(),
    getView: () => ({
      cam: [scene.camera.position.x, scene.camera.position.y, scene.camera.position.z],
      rocket: [scene.rocket.position.x, scene.rocket.position.y, scene.rocket.position.z],
      dist: scene.camera.position.distanceTo(scene.rocket.position),
    }),
    getGym: () => trainer.snap(watch),
    getBrain: () => trainer.brain,
    getPilot: () => hooks.getPilot(),
    getWatch: () => ({
      t: watch.t,
      term: watch.term,
      gen: trainer.watchGen,
      idx: trainer.watchIdx,
      speed: watch.v.len(),
      energy: watch.energy,
      warp: hooks.getWarp(),
      appliedWarp,
    }),
    recycleWatch: () => recycleWatch(true),
    resetNet: () => applyNetReset(),
  };
  const w = window as unknown as { __controlsTest?: typeof probe; __spacey?: typeof probe };
  w.__controlsTest = probe;
  w.__spacey = probe;

  function applyManual(dt: number) {
    const keys = qaKeys.size ? qaKeys : input.keys;
    const a = readActions(keys);
    sim.manual.pitch = a.pitch;
    sim.manual.yaw = a.yaw;
    sim.manual.roll = a.roll;
    if (a.throttleUp) sim.manual.throttle = Math.min(1, sim.manual.throttle + dt * 0.7);
    if (a.throttleDown) sim.manual.throttle = Math.max(0, sim.manual.throttle - dt * 0.7);
    sim.manual.engines = a.engines3 ? 3 : 1;
    if (a.throttleUp) sim.manual.fire = true;
    if (a.throttleDown && sim.manual.throttle < 0.05) sim.manual.fire = false;
  }

  function trainPaused() {
    return hooks.getPaused();
  }

  function soloPaused() {
    return hooks.getPaused() || !hooks.getStarted();
  }

  function stepWatch(wallDt: number) {
    if (watch.terminated()) {
      watchHold += wallDt;
      if (watchHold > 1.2) recycleWatch(true);
      return;
    }
    const warp = Math.max(1, hooks.getWarp());
    appliedWarp = warp;
    watchAcc += Math.max(0, wallDt) * warp;
    let n = 0;
    const maxSteps = 240;
    while (watchAcc >= DT && !watch.terminated() && n < maxSteps) {
      watch.step(DT);
      watchAcc -= DT;
      n += 1;
    }
    if (watchAcc > 4) watchAcc = 4;
  }

  function tickBody(dt: number, rawDt = dt) {
    const mission = hooks.getMission();
    const pilot = hooks.getPilot();
    const seed = hooks.getSeed();
    const epoch = hooks.getBrainEpoch();
    const train = pilot === "train";

    if (epoch !== lastEpoch) {
      lastEpoch = epoch;
      lastSeed = seed;
      lastMission = mission;
      lastPilot = pilot;
      applyNetReset();
    } else if (pilot !== lastPilot) {
      lastPilot = pilot;
      lastMission = mission;
      lastSeed = seed;
      restart();
    } else if (!train && mission !== lastMission) {
      lastMission = mission;
      lastSeed = seed;
      sim = bootSolo();
      scene.resetTrail();
      scene.resetLook();
    } else if (train && mission !== lastMission) {
      lastMission = mission;
    } else if (seed !== lastSeed) {
      lastSeed = seed;
      if (train) recycleWatch(true);
      else {
        sim = bootSolo();
        scene.resetTrail();
        scene.resetLook();
      }
    }

    if (train && !trainPaused()) {
      stepWatch(rawDt);
      const budget = 10;
      const t0 = performance.now();
      let n = 0;
      while (performance.now() - t0 < budget && n < 80) {
        trainer.stepAll(DT);
        n += 1;
        if (trainer.allDone()) {
          const prevE = trainer.brain.energy;
          const elite = trainer.finishGen();
          const unlocked = trainer.unlocked;
          const stage = missionLabel(trainer.brain.energy);
          hooks.onNote(
            unlocked
              ? `Unlocked ${missionLabel(trainer.brain.energy)} · gen ${trainer.brain.gen}`
              : `Gen ${trainer.brain.gen} · ${stage} · land ${(trainer.brain.landRate * 100).toFixed(0)}% · σ ${trainer.brain.sigma.toFixed(2)} · ${elite?.term ?? ""}`,
          );
          hooks.onBrain(trainer.brain);
          trainer.beginGen();
          if (trainer.brain.energy !== prevE) recycleWatch(true);
        }
      }
    } else if (!train && !soloPaused()) {
      const warp = hooks.getWarp();
      const stepsBudget = Math.min(40, 4 * warp);
      acc += dt * warp;
      let n = 0;
      while (acc >= DT && n < stepsBudget) {
        if (pilot === "manual") applyManual(DT);
        sim.step(DT);
        acc -= DT;
        n += 1;
        if (sim.terminated()) break;
      }
      if (acc > DT * 4) acc = 0;
    }

    uiT += dt;
    if (uiT > 0.08) {
      uiT = 0;
      const s = snapshot(viewed());
      hooks.onSnap({
        t: s.t,
        alt: s.alt,
        engineAlt: s.engineAlt,
        speed: s.speed,
        range: s.range,
        tiltDeg: s.tiltDeg,
        fuel: s.fuel,
        phase: s.phase,
        term: s.term,
        qkpa: s.qkpa,
        mach: s.mach,
        throttle: s.throttle,
        nEngines: s.nEngines,
        g: s.g,
        intact: s.intact,
        energy: train ? trainer.brain.energy : s.energy,
        watch: train,
        watchGen: trainer.watchGen,
        watchIdx: trainer.watchIdx,
      });
      if (train) hooks.onGym(trainer.snap(watch));
    }

    scene.sync(viewed(), hooks.getCam(), dt);
    if (train) {
      scene.setSwarm(
        trainer.agents.map((a) => {
          const bx = a.sim.bodyX();
          return {
            trail: a.trail,
            len: Math.min(280, a.trailLen),
            landed: a.landed,
            dead: a.sim.terminated() && !a.landed,
            hero: false,
            px: a.sim.p.x,
            py: a.sim.p.y,
            pz: a.sim.p.z,
            bx: bx.x,
            by: bx.y,
            bz: bx.z,
          };
        }),
        true,
      );
    } else {
      scene.setSwarm([], false);
    }
    scene.render();
  }

  function tick() {
    if (!running) return;
    const now = performance.now();
    const rawDt = (now - last) / 1000;
    last = now;
    const dt = Math.min(0.1, rawDt);
    try {
      tickBody(dt, rawDt);
    } catch (err) {
      console.error(err);
    }
    if (running) requestAnimationFrame(tick);
  }

  requestAnimationFrame(tick);

  return {
    restart,
    sim: () => viewed(),
    dispose() {
      running = false;
      input.dispose();
      scene.dispose();
      const win = window as unknown as { __controlsTest?: unknown; __spacey?: unknown };
      delete win.__controlsTest;
      delete win.__spacey;
    },
  };
}
