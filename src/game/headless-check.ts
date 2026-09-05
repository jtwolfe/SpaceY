import { Sim } from "./sim";

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

function main() {
  const cases = [
    [0, 42],
    [0, 7],
    [0.22, 3],
    [0.68, 5],
    [1, 1],
  ] as const;
  let fail = false;
  for (const [e, s] of cases) {
    const r = trial(e, s);
    console.log(JSON.stringify({ ...r, samples: r.samples.slice(0, 8) }));
    if (e === 0 && r.term !== "landed") fail = true;
  }
  if (fail) process.exit(1);
}

main();
