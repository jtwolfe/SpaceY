import { DT } from "./constants";
import { createInput, readActions } from "./input";
import { snapshot, Sim, type Pilot } from "./sim";
import { SpaceyScene } from "./scene";
import { Trainer, defaultBrain, type Brain, type GymSnap } from "./trainer";
import type { Mission } from "./scenario";
import { energyForMission } from "./scenario";
import type { AppPilot, CamMode, HudSnap } from "./store";
import { N_WEIGHTS } from "./policy";

export type RuntimeHooks = {
  getPilot: () => AppPilot;
  getMission: () => Mission;
  getCam: () => CamMode;
  getWarp: () => number;
  getPaused: () => boolean;
  getStarted: () => boolean;
  getSeed: () => number;
  getBrain: () => Brain;
  onSnap: (s: HudSnap) => void;
  onBrain: (b: Brain) => void;
  onGym: (g: GymSnap) => void;
  onNote: (n: string) => void;
};

export function createRuntime(canvas: HTMLCanvasElement, hooks: RuntimeHooks) {
  const scene = new SpaceyScene(canvas);
  const input = createInput();
  const qaKeys = new Set<string>();
  const loaded = hooks.getBrain();
  const trainer = new Trainer({
    ...defaultBrain(),
    ...loaded,
    sigma: Math.max(0.12, loaded.sigma || 0.2),
    weights: loaded.weights?.length === N_WEIGHTS ? [...loaded.weights] : defaultBrain().weights,
  });
  trainer.brain.energy = energyForMission(hooks.getMission());
  trainer.beginGen();

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
  let uiT = 0;

  function energyNow() {
    if (hooks.getPilot() === "train") return trainer.brain.energy;
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

  function restart() {
    if (hooks.getPilot() === "train") {
      trainer.brain.energy = energyForMission(hooks.getMission());
      trainer.beginGen();
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
    }),
    recycleWatch: () => recycleWatch(true),
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

  function stepWatch(dt: number) {
    if (watch.terminated()) {
      watchHold += dt;
      if (watchHold > 1.2) recycleWatch(true);
      return;
    }
    watchAcc += Math.min(0.25, dt);
    while (watchAcc >= DT && !watch.terminated()) {
      watch.step(DT);
      watchAcc -= DT;
    }
    if (watchAcc > DT * 4) watchAcc = 0;
  }

  function tickBody(dt: number, rawDt = dt) {
    const mission = hooks.getMission();
    const pilot = hooks.getPilot();
    const seed = hooks.getSeed();
    const train = pilot === "train";

    if (mission !== lastMission || pilot !== lastPilot) {
      lastMission = mission;
      lastPilot = pilot;
      lastSeed = seed;
      if (train) trainer.brain.energy = energyForMission(mission);
      restart();
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
          const elite = trainer.finishGen();
          hooks.onNote(
            `Gen ${trainer.brain.gen} · land ${(trainer.brain.landRate * 100).toFixed(0)}% · σ ${trainer.brain.sigma.toFixed(2)} · ${elite?.term ?? ""}`,
          );
          hooks.onBrain(trainer.brain);
          trainer.beginGen();
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
        energy: s.energy,
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
