/** GNC-only RTLS tape. Not imported by the app. */
import { Sim } from "./sim";
import { spawnAt } from "./scenario";

function tape(energy: number, seed: number, dtSample: number) {
  const sim = Sim.start(energy, seed, { destroy: energy > 0.3, pilot: "autopilot" });
  const sp = spawnAt(energy, seed);
  const lines: string[] = [];
  const tEnd = sim.timeout;
  let next = 0;
  let maxQ = 0;
  let maxAoa = 0;
  let minR = sim.nav.rangeH;
  let maxVh = Math.hypot(sim.v.x, sim.v.y);
  let nLit = 0;
  let entryT = 0;
  let peakN = 0;
  while (sim.t < tEnd && !sim.terminated()) {
    sim.step(0.02);
    maxQ = Math.max(maxQ, sim.lastQ);
    maxAoa = Math.max(maxAoa, Math.abs(sim.lastAoa));
    minR = Math.min(minR, sim.nav.rangeH);
    maxVh = Math.max(maxVh, Math.hypot(sim.v.x, sim.v.y));
    if (sim.lastN > 0) nLit += 1;
    if (sim.entryBurn === "on") entryT += 0.02;
    peakN = Math.max(peakN, sim.lastN);
    if (sim.t >= next) {
      const bx = sim.bodyX();
      const vh = Math.hypot(sim.v.x, sim.v.y);
      const r = Math.max(1, sim.nav.rangeH);
      const engPad = (sim.p.x * bx.x + sim.p.y * bx.y) / r;
      const gamma = (Math.atan2(-sim.v.z, Math.max(1, vh)) * 180) / Math.PI;
      lines.push(
        `${sim.t.toFixed(0)}s z=${sim.p.z.toFixed(0)} r=${sim.nav.rangeH.toFixed(0)} v=${sim.nav.speed.toFixed(0)} vh=${vh.toFixed(0)} vd=${(-sim.v.z).toFixed(0)} g=${gamma.toFixed(0)} ph=${sim.phase} n=${sim.lastN} thr=${sim.lastThrottle.toFixed(2)} q=${(sim.lastQ / 1000).toFixed(1)} aoa=${((sim.lastAoa * 180) / Math.PI).toFixed(1)} tilt=${((sim.nav.tilt * 180) / Math.PI).toFixed(0)} fuel=${sim.fuel.toFixed(0)} entry=${sim.entryBurn} engPad=${engPad.toFixed(2)}`,
      );
      next += dtSample;
    }
  }
  return {
    energy,
    seed,
    spawnR: +Math.hypot(sp.p.x, sp.p.y).toFixed(0),
    spawnV: +sp.v.len().toFixed(0),
    term: sim.term,
    destroy: sim.destroyReason,
    t: +sim.t.toFixed(1),
    speed: +sim.nav.speed.toFixed(1),
    range: +sim.nav.rangeH.toFixed(1),
    minR: +minR.toFixed(1),
    alt: +sim.nav.engineAlt.toFixed(1),
    fuel: +sim.fuel.toFixed(0),
    lights: sim.engine.lights,
    nLit,
    entryT: +entryT.toFixed(1),
    peakN,
    maxQ: +(maxQ / 1000).toFixed(1),
    maxAoa: +((maxAoa * 180) / Math.PI).toFixed(1),
    tilt: +((sim.nav.tilt * 180) / Math.PI).toFixed(1),
    lines,
  };
}

function main() {
  console.log("--- pad 8 ---");
  let padLand = 0;
  for (let i = 0; i < 8; i++) {
    const r = tape(0, 42 + i * 11, 8);
    if (r.term === "landed") padLand += 1;
    console.log(JSON.stringify({ seed: r.seed, term: r.term, t: r.t, range: r.range, speed: r.speed, lights: r.lights }));
  }
  console.log("padLand", padLand, "/8");

  console.log("--- 2km 8 ---");
  let slamLand = 0;
  for (let i = 0; i < 8; i++) {
    const r = tape(0.22, 3 + i * 19, 8);
    if (r.term === "landed") slamLand += 1;
    console.log(JSON.stringify({ seed: r.seed, term: r.term, t: r.t, range: r.range, minR: r.minR, speed: r.speed, lights: r.lights }));
  }
  console.log("slamLand", slamLand, "/8");

  console.log("--- glide 8 ---");
  let glideLand = 0;
  const glideCounts: Record<string, number> = {};
  for (let i = 0; i < 8; i++) {
    const r = tape(0.68, 5 + i * 17, 20);
    if (r.term === "landed") glideLand += 1;
    glideCounts[r.term] = (glideCounts[r.term] ?? 0) + 1;
    console.log(JSON.stringify({ seed: r.seed, term: r.term, destroy: r.destroy, t: r.t, range: r.range, minR: r.minR, speed: r.speed, lights: r.lights, maxQ: r.maxQ, maxAoa: r.maxAoa }));
  }
  console.log("glideLand", glideLand, "/8", glideCounts);

  console.log("--- rtls hero seed 1 ---");
  const hero = tape(1, 1, 8);
  for (const l of hero.lines) console.log(l);
  console.log(
    JSON.stringify({
      term: hero.term,
      destroy: hero.destroy,
      t: hero.t,
      range: hero.range,
      minR: hero.minR,
      speed: hero.speed,
      alt: hero.alt,
      fuel: hero.fuel,
      lights: hero.lights,
      entryT: hero.entryT,
      maxQ: hero.maxQ,
      maxAoa: hero.maxAoa,
      tilt: hero.tilt,
    }),
  );

  console.log("--- rtls 16 seeds ---");
  const counts: Record<string, number> = {};
  const dest: Record<string, number> = {};
  let land = 0;
  const rows: unknown[] = [];
  for (let i = 0; i < 16; i++) {
    const r = tape(1, 1 + i * 13, 0);
    counts[r.term] = (counts[r.term] ?? 0) + 1;
    dest[r.destroy] = (dest[r.destroy] ?? 0) + 1;
    if (r.term === "landed") land += 1;
    rows.push({
      seed: r.seed,
      term: r.term,
      destroy: r.destroy,
      t: r.t,
      range: r.range,
      minR: r.minR,
      speed: r.speed,
      alt: r.alt,
      fuel: r.fuel,
      lights: r.lights,
      entryT: r.entryT,
      maxAoa: r.maxAoa,
      maxQ: r.maxQ,
      tilt: r.tilt,
    });
  }
  console.log(JSON.stringify({ counts, dest, land, rows }, null, 2));
}

main();
