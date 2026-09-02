#!/usr/bin/env node
// Compare a wasm32 LEO nominal against the documented success box.
// Usage:
//   wasm-bindgen --target nodejs --out-dir target/leo-wasm-check --out-name spacey \
//     target/wasm32-unknown-unknown/release/spacey_sim.wasm
//   node scripts/leo_wasm_check.mjs [seed=88]
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
// Node target bindgen is CommonJS — keep it out of web/ (type: module).
const dir = process.env.LEO_WASM_DIR ?? resolve(here, "../target/leo-wasm-check");
const require = createRequire(import.meta.url);
const { run_leo_nominal } = require(resolve(dir, "spacey.js"));

const seed = Number(process.argv[2] ?? 88) >>> 0;
const snap = JSON.parse(run_leo_nominal(seed, true, 0.0, 140000));
const line = [
  `seed=${seed}`,
  `term=${JSON.stringify(snap.term)}`,
  `success=${snap.success}`,
  `phase=${snap.phase}`,
  `t=${snap.t?.toFixed?.(2)}`,
  `alt=${snap.alt?.toFixed?.(1)}`,
  `spd=${snap.speed?.toFixed?.(2)}`,
  `range_h=${snap.range_h?.toFixed?.(1)}`,
  `range_gc=${snap.range_gc?.toFixed?.(1)}`,
  `fuel=${snap.fuel?.toFixed?.(1)}`,
  `tilt=${snap.tilt_deg?.toFixed?.(1)}`,
  `intact=${snap.intact}`,
  `destroy=${JSON.stringify(snap.destroy_reason)}`,
  `alt_km=${(snap.alt / 1000).toFixed(2)}`,
  `vin=${(snap.speed_inertial / 1000).toFixed(3)}`,
].join(" ");
console.log(line);

if (!snap.success) {
  console.error("wasm32 LEO nominal did not hit TermReason::Success");
  process.exit(1);
}
