#!/usr/bin/env node
// wasm32 pad gym: a zero-weight policy must terminate without landing.
// Usage:
//   wasm-bindgen --target nodejs --out-dir target/wasm-gym-check --out-name spacey \
//     target/wasm32-unknown-unknown/release/spacey_sim.wasm
//   node scripts/wasm_gym_check.mjs [seed=1]
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const dir = process.env.WASM_GYM_DIR ?? resolve(here, "../target/wasm-gym-check");
const require = createRequire(import.meta.url);
const { run_hover_policy } = require(resolve(dir, "spacey.js"));

const seed = Number(process.argv[2] ?? 1) >>> 0;
const snap = JSON.parse(run_hover_policy(seed, true, 0.0, 20000));
const line = [
  `seed=${seed}`,
  `scenario=${JSON.stringify(snap.scenario)}`,
  `pilot=${JSON.stringify(snap.pilot)}`,
  `term=${JSON.stringify(snap.term)}`,
  `success=${snap.success}`,
  `terminated=${snap.terminated}`,
  `n_engines=${snap.n_engines}`,
  `t=${snap.t?.toFixed?.(2)}`,
  `alt=${snap.alt?.toFixed?.(1)}`,
  `spd=${snap.speed?.toFixed?.(2)}`,
  `intact=${snap.intact}`,
].join(" ");
console.log(line);

if (snap.scenario !== "pad") {
  console.error("expected pad scenario");
  process.exit(1);
}
if (snap.pilot !== "policy") {
  console.error("expected policy pilot");
  process.exit(1);
}
if (!snap.terminated) {
  console.error("pad policy did not terminate");
  process.exit(1);
}
if (snap.success) {
  console.error("zero-weight policy must not hit the success box");
  process.exit(1);
}
if (snap.n_engines !== 0) {
  console.error("zero-weight policy must not light engines");
  process.exit(1);
}
