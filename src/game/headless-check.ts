import { Sim } from "./sim";
import { spawnAt, SLAM_SPAWN_RADIUS_M, SLAM_SPAWN_MIN_R_M } from "./scenario";
import {
  N_HIDDEN,
  N_HIDDEN_LAYERS_MAX,
  N_IN,
  N_OUT,
  growWeights,
  keepSinking,
  layersFromLen,
  mlpForward,
  nWeights,
  residualLive,
  topologyOf,
  zeroWeights,
} from "./policy";
import { goalFromState, planRef } from "./reftraj";
import { Vec3 } from "./math";
import { GATE_RATE, MU, POP, Trainer, defaultBrain, layersForEnergy, scoreSim, selectParents, type Agent } from "./trainer";
import { attitudeCommand, emptyControls, landingEngineCluster, makeNav, NOMINAL_GAINS, unpoweredLandingAim } from "./guidance";
import { finQEnable } from "./vehicle";

function trial(energy: number, seed: number) {
  const sim = Sim.start(energy, seed, { destroy: energy > 0.3, pilot: "autopilot" });
  const samples: string[] = [];
  const tEnd = sim.timeout;
  let next = 0;
  while (sim.t < tEnd && !sim.terminated()) {
    sim.step(0.02);
    if (sim.t >= next) {
      samples.push(
        `${sim.t.toFixed(0)}s z=${sim.p.z.toFixed(0)} r=${sim.nav.rangeH.toFixed(0)} v=${sim.nav.speed.toFixed(0)} ph=${sim.phase} n=${sim.lastN} thr=${sim.lastThrottle.toFixed(2)} q=${(sim.lastQ / 1000).toFixed(1)}`,
      );
      next += energy >= 0.6 ? 20 : 2;
    }
  }
  return {
    energy,
    seed,
    term: sim.term,
    t: +sim.t.toFixed(1),
    speed: +sim.nav.speed.toFixed(2),
    range: +sim.nav.rangeH.toFixed(1),
    tiltDeg: +((sim.nav.tilt * 180) / Math.PI).toFixed(1),
    alt: +sim.nav.engineAlt.toFixed(1),
    fuel: +sim.fuel.toFixed(0),
    lights: sim.engine.lights,
    destroy: sim.destroyReason,
    samples,
  };
}

function fakeGen(tr: Trainer, nLand: number) {
  tr.beginGen();
  tr.agents.forEach((a, i) => {
    a.landed = i < nLand;
    a.term = i < nLand ? "landed" : "miss";
    a.fit = i < nLand ? 12_000 - i : -100 - i;
  });
  tr.genLands = nLand;
  tr.doneCount = POP;
  tr.finishGen();
}

function main() {
  const cases = [
    [0, 42],
    [0, 7],
    [0.22, 3],
    [0.68, 5],
    [1, 1],
  ] as const;
  let fail = false;
  const why: string[] = [];
  const mark = (ok: boolean, msg: string) => {
    if (!ok) {
      fail = true;
      why.push(msg);
    }
  };
  for (const [e, s] of cases) {
    const r = trial(e, s);
    console.log(JSON.stringify({ ...r, samples: r.samples.slice(0, 8) }));
    if (e === 0 && r.term !== "landed") fail = true;
    if (e === 0 && r.lights > 1) fail = true;
    if (e === 0.22) {
      for (const line of r.samples.slice(0, 4)) if (/n=[1-9]/.test(line)) fail = true;
      if (r.samples.some((line) => /n=3/.test(line) && /thr=0\.4/.test(line))) fail = true;
    }
    if (e === 0.22 && r.lights > 1) fail = true;
  }
  const ranges: number[] = [];
  for (let i = 0; i < 16; i++) {
    const s = spawnAt(0.22, 100 + i * 19);
    const r = Math.hypot(s.p.x, s.p.y);
    ranges.push(r);
    if (r < SLAM_SPAWN_MIN_R_M - 1 || r > SLAM_SPAWN_RADIUS_M + 1) fail = true;
  }
  const spread = Math.max(...ranges) - Math.min(...ranges);
  if (spread < 20) fail = true;
  const pad = spawnAt(0, 42);
  if (Math.hypot(pad.p.x, pad.p.y) > 25) fail = true;
  console.log(JSON.stringify({ slamRanges: ranges.map((n) => +n.toFixed(1)), spread: +spread.toFixed(1) }));
  if (nWeights(1) !== 266 || nWeights(4) !== 482 || nWeights(6) !== 626 || N_IN !== 21) fail = true;
  if (POP !== 64 || MU !== 32) fail = true;
  const z = mlpForward(zeroWeights(), Array.from({ length: N_IN }, () => 0));
  if (z.layers !== 1 || z.h.length !== N_HIDDEN || z.y.length !== N_OUT) fail = true;
  if (Array.from(z.y).some((v) => v !== 0) || Array.from(z.h).some((v) => v !== 0)) fail = true;
  const x = Array.from({ length: N_IN }, (_, i) => (i - 10) * 0.07);
  const w0 = zeroWeights(1).map((_, i) => ((i * 17) % 100) / 200 - 0.25);
  const y0 = Array.from(mlpForward(w0, x).y);
  let wg = w0;
  for (let L = 2; L <= N_HIDDEN_LAYERS_MAX; L++) {
    wg = growWeights(wg);
    if (layersFromLen(wg.length) !== L) fail = true;
    const yg = Array.from(mlpForward(wg, x).y);
    if (y0.some((v, i) => Math.abs(v - yg[i]) > 1e-9)) fail = true;
  }
  let refOk = 0;
  const refNotes: { seed: number; ok: boolean; range: number; speed: number; tLight: number }[] = [];
  for (let i = 0; i < 12; i++) {
    const sp = spawnAt(0.22, 100 + i * 19);
    const windDir = ((100 + i * 19) % 360) * (Math.PI / 180);
    const trj = planRef({
      p: sp.p,
      v: sp.v,
      fuel: sp.fuel,
      timeout: sp.timeout,
      windScale: sp.wind,
      windDir,
    });
    if (trj.ok) refOk += 1;
    refNotes.push({ seed: 100 + i * 19, ok: trj.ok, range: +trj.range.toFixed(1), speed: +trj.speed.toFixed(1), tLight: +trj.tLight.toFixed(1) });
  }
  if (refOk < 1) fail = true;

  const gCoast = goalFromState({
    p: new Vec3(80, 0, 2050),
    v: new Vec3(28, 0, -120),
    fuel: 8500,
    lit: false,
  });
  if (!(gCoast.tLight > 0.5) || gCoast.goVertical || gCoast.dir.z > 0.98) fail = true;
  const gPad = goalFromState({
    p: new Vec3(8, 4, 40),
    v: new Vec3(1, 0.4, -12),
    fuel: 2200,
    lit: true,
    prev: gCoast,
  });
  if (!gPad.goVertical || gPad.dir.z < 0.85 || Math.abs(gPad.tLight) > 0.5) fail = true;
  const gYank = goalFromState({
    p: new Vec3(-80, 0, 400),
    v: new Vec3(-20, 0, -80),
    fuel: 4000,
    lit: true,
    prev: gCoast,
  });
  if (gYank.dir.dot(gCoast.dir) < 0.7) fail = true;
  const st = Sim.start(0.22, 3, { destroy: false, pilot: "student", weights: zeroWeights() });
  st.step(0.02);
  if (!st.goal || !Number.isFinite(st.lastRefDist)) fail = true;

  if (finQEnable(2_000) !== 0 || finQEnable(9_000) !== 1 || Math.abs(finQEnable(6_000) - 0.5) > 1e-9) fail = true;
  const up = new Vec3(0, 0, 1);
  const by = new Vec3(0, 1, 0);
  const bz = new Vec3(-1, 0, 0);
  const want = new Vec3(0.35, 0.1, 0.93).normalized();
  const attHi = attitudeCommand(up, by, bz, new Vec3(), want, "landing", 7_500, NOMINAL_GAINS);
  const attLo = attitudeCommand(up, by, bz, new Vec3(), want, "landing", 1_000, NOMINAL_GAINS);
  const finHi = Math.hypot(attHi.finP, attHi.finY, attHi.finR);
  const finLo = Math.hypot(attLo.finP, attLo.finY, attLo.finR);
  if (!(finHi > 0.02) || finLo > 1e-9) fail = true;

  const hop = Sim.start(0.22, 3, { destroy: false, pilot: "autopilot" });
  hop.stepFor(0.6);
  const hopFin = Math.hypot(hop.lastFins[0], hop.lastFins[1], hop.lastFins[2]);
  if (hop.lastQ < 5_000 || hopFin < 0.01 || hop.coastLatched) fail = true;
  if (hop.spawnRange < 30) fail = true;
  if (hop.lastN > 0) fail = true;

  const aimP = new Vec3(80, 0, 2050);
  const aimV = new Vec3(28, 0, -120);
  const aimNav = makeNav(aimP, aimV, new Vec3(0, 0, 1), 8_000, 0.1, 0.4, 8500, 34_000);
  const aim = unpoweredLandingAim(aimNav);
  const aimR = Math.hypot(aimP.x, aimP.y);
  const noseAtPad = (-aimP.x * aim.x + -aimP.y * aim.y) / aimR;
  const enginesAtPad = (aimP.x * aim.x + aimP.y * aim.y) / aimR;
  if (noseAtPad > 0.05 || enginesAtPad < 0 || aim.z < 0.85) fail = true;

  const upNav = new Vec3(0, 0, 1);
  const padCl = landingEngineCluster(makeNav(new Vec3(8, 0, 80), new Vec3(0, 0, -20), upNav, 400, 0, 0.1, 2800, 28_400));
  if (padCl.n !== 1 || padCl.use3) fail = true;
  const hopCl = landingEngineCluster(makeNav(new Vec3(80, 0, 920), new Vec3(28, 0, -150), upNav, 8_000, 0.1, 0.4, 8500, 34_100));
  if (hopCl.n !== 1 || hopCl.use3 || hopCl.throttle < 0.7) fail = true;
  const floorCl = landingEngineCluster(makeNav(new Vec3(50, 0, 430), new Vec3(0, 0, -122), upNav, 4_000, 0.05, 0.3, 8500, 34_100));
  if (floorCl.n !== 1 || floorCl.use3 || floorCl.throttle < 0.99) fail = true;
  const hotCl = landingEngineCluster(makeNav(new Vec3(40, 0, 180), new Vec3(10, 0, -200), upNav, 5_000, 0.1, 0.5, 8500, 34_100));
  if (hotCl.n !== 3 || hotCl.throttle < 0.55) fail = true;

  const climbNav = makeNav(new Vec3(8, 0, 200), new Vec3(0, 0, 2.4), upNav, 400, 0, 0.1, 2800, 28_400);
  const uClimb = emptyControls();
  uClimb.nEngines = 1;
  uClimb.throttle = 1;
  keepSinking(uClimb, climbNav);
  if (uClimb.nEngines !== 1 || Math.abs(uClimb.throttle - 0.4) > 1e-9) fail = true;
  const uDeck = emptyControls();
  uDeck.nEngines = 1;
  uDeck.throttle = 1;
  const deckNav = makeNav(new Vec3(0, 0, 28), new Vec3(0, 0, 2.4), upNav, 400, 0, 0.1, 2800, 28_400);
  keepSinking(uDeck, deckNav);
  if (uDeck.nEngines !== 0 || uDeck.throttle !== 0) fail = true;

  const stay = Sim.start(0, 42, { destroy: false, pilot: "autopilot" });
  let sawLight = false;
  let midAirShut = false;
  while (stay.t < 40 && !stay.terminated()) {
    stay.step(0.02);
    if (stay.landingIgnited && stay.lastN > 0) sawLight = true;
    if (sawLight && stay.nav.engineAlt > 8 && stay.lastN === 0) midAirShut = true;
  }
  if (stay.term !== "landed" || !sawLight || midAirShut) fail = true;

  hop.stepFor(3.4);
  if (hop.lastN > 0 || hop.coastLatched) fail = true;
  const hopBx = hop.bodyX();
  const hopR = hop.nav.rangeH;
  const hopNose = hopR > 1 ? (-hop.p.x * hopBx.x + -hop.p.y * hopBx.y) / hopR : 0;
  if (hopNose > 0.35) fail = true;

  const padNow = Sim.start(0, 42, { destroy: false, pilot: "autopilot" });
  if (!padNow.coastLatched || padNow.nav.engineAlt > 400) fail = true;
  padNow.stepFor(0.3);
  if (padNow.lastQ > 3_500) fail = true;

  const land = Sim.start(0, 42, { destroy: false, pilot: "autopilot" });
  land.stepFor(40);
  if (land.term !== "landed") fail = true;
  const landBase = scoreSim(land);
  land.lateTiltT = 4;
  land.lateVhT = 4;
  land.coastRemain = 120;
  land.coastVh = 40;
  land.coastKill = 0;
  land.coastPredKill = 0;
  land.coastLingerT = 40_000;
  land.coastLatched = true;
  land.engineOnT = 18;
  land.thrustAwayShortT = 4;
  land.threeOnT = 6;
  land.threeMinT = 4;
  const landTaxed = scoreSim(land);
  const miss = Sim.start(0.22, 3, { destroy: false, pilot: "student", weights: zeroWeights() });
  miss.stepFor(90);
  const missFit = scoreSim(miss);
  if (!(landTaxed > missFit) || !(landBase > landTaxed) || landTaxed < 1_000) fail = true;

  const tr = new Trainer();
  tr.beginGen();
  if (tr.agents.length !== POP || tr.brain.weights.length !== 266) fail = true;
  if (tr.agents[0].z.some((v) => v !== 0)) fail = true;
  tr.agents.forEach((a, i) => {
    a.fit = i === 0 ? 100 : -i;
    a.term = "miss";
  });
  tr.finishGen();
  if (tr.brain.sigma < 0.05 || tr.brain.sigma > 0.28) fail = true;
  if (tr.brain.weights.length !== 266) fail = true;

  const yMean = Array.from(mlpForward(tr.brain.weights, x).y);
  if (!tr.growLayer()) fail = true;
  if (tr.brain.weights.length !== 338) fail = true;
  if (tr.brain.ps.length !== 338 || tr.brain.pc.length !== 338 || tr.brain.diagC.length !== 338) fail = true;
  const yGrown = Array.from(mlpForward(tr.brain.weights, x).y);
  if (yMean.some((v, i) => Math.abs(v - yGrown[i]) > 1e-9)) fail = true;
  tr.growLayer();
  tr.growLayer();
  tr.growLayer();
  tr.growLayer();
  if (tr.brain.weights.length !== 626) fail = true;
  if (tr.growLayer()) fail = true;

  const gate = new Trainer();
  const gateLands = Math.ceil(GATE_RATE * POP);
  fakeGen(gate, gateLands);
  const after1 = { energy: gate.brain.energy, n: gate.brain.weights.length, streak: gate.streak };
  fakeGen(gate, gateLands);
  const after2 = { energy: gate.brain.energy, n: gate.brain.weights.length, grown: gate.grown };
  if (after1.energy !== 0 || after1.n !== 266) fail = true;
  if (Math.abs(after2.energy - 0.22) > 1e-9) fail = true;
  if (after2.n !== 338) fail = true;
  if (!after2.grown) fail = true;
  if (layersForEnergy(0) !== 1 || layersForEnergy(0.22) !== 2 || layersForEnergy(0.68) !== 3 || layersForEnergy(1) !== 5) fail = true;
  const b2 = defaultBrain();
  b2.energy = 0.22;
  const fitted = new Trainer(b2);
  if (fitted.brain.weights.length !== 338) fail = true;
  const yFit0 = Array.from(mlpForward(zeroWeights(1), x).y);
  const yFit1 = Array.from(mlpForward(fitted.brain.weights, x).y);
  if (yFit0.some((v, i) => Math.abs(v - yFit1[i]) > 1e-9)) fail = true;

  const bRtls = defaultBrain();
  bRtls.energy = 1;
  const fittedRtls = new Trainer(bRtls);
  mark(fittedRtls.brain.weights.length === 554, `rtlsFitLen ${fittedRtls.brain.weights.length}`);
  mark(layersFromLen(fittedRtls.brain.weights.length) === 5, "rtlsFitLayers");

  fakeGen(gate, gateLands);
  fakeGen(gate, gateLands);
  mark(Math.abs(gate.brain.energy - 0.68) <= 1e-9, `glideUnlock ${gate.brain.energy}`);
  fakeGen(gate, gateLands);
  fakeGen(gate, gateLands);
  mark(Math.abs(gate.brain.energy - 1) <= 1e-9, `rtlsUnlock ${gate.brain.energy}`);
  const afterRtls = new Trainer(gate.brain);
  mark(afterRtls.brain.weights.length === 554, `afterRtls ${afterRtls.brain.weights.length}`);

  function stubAgent(landed: boolean, fit: number, minRange: number): Agent {
    return {
      weights: [],
      z: [],
      sim: { minRange } as Agent["sim"],
      fit,
      term: landed ? "landed" : "miss",
      landed,
      trail: new Float32Array(0),
      trailLen: 0,
      trailCursor: 0,
    };
  }
  const farMisses = Array.from({ length: 40 }, (_, i) => stubAgent(false, 8_000 - i, 20_000 + i * 10));
  const oneLand = [stubAgent(true, 12_000, 2), ...farMisses];
  const padPool = selectParents(oneLand, 0);
  mark(padPool.length === MU, `padPool ${padPool.length}`);
  mark(padPool.some((a) => !a.landed && a.sim.minRange > 1_000), "padPoolFar");
  const rtlsPool = selectParents(oneLand, 1);
  mark(rtlsPool.length === 1 && !!rtlsPool[0].landed, `rtlsPool ${rtlsPool.length} land=${rtlsPool[0]?.landed}`);
  const noLand = farMisses.map((a, i) => stubAgent(false, 9_000 - i, i === 3 ? 80 : 18_000 + i));
  const rtlsClose = selectParents(noLand, 1);
  mark(rtlsClose.length === 1 && rtlsClose[0].sim.minRange === 80, `rtlsClose ${rtlsClose.length} r=${rtlsClose[0]?.sim.minRange}`);
  const allFar = farMisses.map((a, i) => stubAgent(false, 9_000 - i, 15_000 + i * 100));
  const rtlsFar = selectParents(allFar, 1);
  mark(rtlsFar.length === 8, `rtlsFar ${rtlsFar.length}`);
  mark(!rtlsFar.some((a, i) => i > 0 && a.sim.minRange < rtlsFar[i - 1].sim.minRange), "rtlsFarSort");
  mark(rtlsFar[0].sim.minRange === Math.min(...allFar.map((a) => a.sim.minRange)), "rtlsFarMin");

  mark(residualLive(0, "entry", 80_000, 2_000, true), "padLiveEntry");
  mark(residualLive(0, "landing", 200, 20, false), "padLiveLand");
  mark(!residualLive(1, "entry", 80_000, 2_000, true), "rtlsFreezeEntry");
  mark(!residualLive(1, "glide", 40_000, 800, false), "rtlsFreezeHiGlide");
  mark(residualLive(1, "landing", 400, 40, false), "rtlsLiveLand");
  mark(residualLive(1, "glide", 8_000, 200, false), "rtlsLiveLoGlide");

  const wild1 = zeroWeights(1).map(() => 2.5);
  const wildR = zeroWeights(5).map(() => 2.5);
  const padStu = Sim.start(0, 42, { destroy: false, pilot: "student", weights: wild1 });
  const padApo = Sim.start(0, 42, { destroy: false, pilot: "autopilot" });
  padStu.stepFor(7);
  padApo.stepFor(7);
  const padDiv = Math.abs(padStu.p.x - padApo.p.x) + Math.abs(padStu.p.z - padApo.p.z);
  mark(padDiv >= 0.4, `padDiv ${padDiv.toFixed(3)}`);
  const rtlsStu = Sim.start(1, 1, { destroy: false, pilot: "student", weights: wildR });
  const rtlsApo = Sim.start(1, 1, { destroy: false, pilot: "autopilot" });
  rtlsStu.stepFor(25);
  rtlsApo.stepFor(25);
  const rtlsDiv = Math.abs(rtlsStu.p.x - rtlsApo.p.x) + Math.abs(rtlsStu.p.z - rtlsApo.p.z);
  mark(rtlsDiv <= 2, `rtlsDiv ${rtlsDiv.toFixed(3)} ph=${rtlsStu.phase}/${rtlsApo.phase} entry=${rtlsStu.entryBurn}/${rtlsApo.entryBurn}`);
  mark(rtlsStu.entryBurn === rtlsApo.entryBurn, "rtlsEntryMatch");

  console.log(
    JSON.stringify({
      nWeights: nWeights(1),
      topology: topologyOf(1),
      max: topologyOf(N_HIDDEN_LAYERS_MAX),
      rtlsLayers: layersForEnergy(1),
      sigma: +tr.brain.sigma.toFixed(3),
      mu: MU,
      unlockGrow: after2,
      rungFit: fitted.brain.weights.length,
      refOk,
      refNotes,
      finHi: +finHi.toFixed(3),
      hopQ: +(hop.lastQ / 1000).toFixed(2),
      hopFin: +hopFin.toFixed(3),
      landTaxed: +landTaxed.toFixed(0),
      missFit: +missFit.toFixed(0),
      fail,
      why,
    }),
  );
  if (fail) process.exit(1);
}

main();
