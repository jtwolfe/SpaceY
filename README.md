# SpaceY

Browser gym for a **Falcon 9-class first stage**: 6DOF physics, grid-fin aero, cold-gas RCS, Merlin 1/3, and an in-browser **CMA-NeuroES** trainer.

The mission is **RTLS**: recover the booster to LZ-1. Training starts as a 2D pad slam and auto-promotes through 2 km, 6DOF, wind, glide, then the full ~80 km RTLS start. The net sees only pad-frame pose, velocity, rates, and fuel, and writes throttle, gimbal, grid fins, engine cluster, and RCS.

Physics and training run in **Rust compiled to WASM**. The scene is **TypeScript + Three.js**.

**Grok Build, the `grok` CLI, grok TUI, and any Grok Build / Grok coding-agent workflow are not part of this project.**

Numbers below are **approximate public figures**, not live telemetry and not a proprietary aero database.

## Run (one command after the usual Rust + Node install)

Prerequisites: Rust (stable) with `wasm32-unknown-unknown`, [`wasm-bindgen-cli`](https://crates.io/crates/wasm-bindgen) **0.2.100**, and Node.js 20+.

```bash
cargo install wasm-bindgen-cli --version 0.2.100 --locked
rustup target add wasm32-unknown-unknown
unset CARGO_TARGET_DIR
npm start
```

That builds the WASM crate and serves the app at [http://localhost:5173](http://localhost:5173).

| Command        | What it does                                      |
|----------------|---------------------------------------------------|
| `npm test`     | Native Rust unit tests                            |
| `npm run build`| Production static build in `web/dist`             |

If `CARGO_TARGET_DIR` is set, bindgen can pick a stale wasm. Unset it so bindgen uses `target/wasm32-unknown-unknown/release/spacey_sim.wasm`.

## What the sim models

- **6DOF rigid body** in ECI, WGS-84 + J2, rotating Earth.
- **US Standard Atmosphere 1976** from sea level through 86 km, exponential tail above.
- **Three actuator families**
  - **Merlins**: 1 or 3 engines (policy cluster output), 40–100% throttle, 2-axis gimbal with slew, Isp mix SL/vac. Once lit, a burn lasts ≥2.5 s; a restart waits ≥6 s. Switching 1↔3 while lit is not a relight.
  - **Grid fins**: four lattices at the interstage, X-mixer, axial drag even at δ=0, weathercock from `r × F` (not a lumped CP hack), hinge-rate limit.
  - **RCS**: policy-owned cold-gas commands (Q-fade, no inner PD). Off during 2D pad/2 km (plane-lock); 6DOF and vacuum can use it. Autopilot demo still uses its own rate loop.
- **Wind / weather**: Florida-east-coast caricature + OU gusts. Training samples **intensity and heading** every episode (stage-scaled). Pad stays nearly calm (≲0.25× the slider); real breeze starts at the Wind stage, and pad mixes two windier gens before promoting. Storm/shear HUD boxes **pin** that bit on; the wind slider is the nominal intensity (default **1×**). Wind is not an MLP input.
- **Destruction** (default on): max-Q, over-G, q-alpha, AoA, spin, hard impact.
- **Success**: ground contact, engine ≲12 m, ≲8 m/s, ≲4 m/s horizontal, within 20 m of the pad, ≲8° tilt, intact. Hovering in the volume is not a land. Fuel-out, hang, or breakup still in the air is scored like refusing to come down; a sit-down on the pad is the cheapest miss.

## How training works

The **policy is the controller**. A small tanh MLP (14 → 8 hidden → 10 actions, 210 weights) reads asinh pad-ENU engine position, velocity, body→ENU quaternion, body rate, and fuel / 40 t. It writes throttle (off or 40–100%), two-axis TVC, three fin axes, 1-or-3 engines, and three RCS axes. Actions are held at **10 Hz**. The plant latches ignition (min burn / restart delay); extra relights cost fitness. No inner attitude PD, no wind/q/Mach/phase inputs, no reference trajectory in the observation.

Weights are evolved with **CMA-ES** (CMA-NeuroES). No backprop. If fitness plateaus, one hidden unit is added (CMA-TWEANN style, new weights start at 0, covariance reset). Cap is 24 hidden so the browser covariance stays small. The champion is saved to **localStorage** so a reload does not wipe a session.

Zero weights keep engines and RCS **off**. CMA has to learn to light a Merlin. Promoting a stage **keeps** the champion weights, runs two mixed generations, then resets covariance.

**Autopilot demo** is the old hand-written tracker on a full RTLS start. It is not in the training loop.

Curriculum is internal. After three generations at ≥30% true lands, pad mixes two gens of a lower/windier pad, then advances. Later stages mix the next stage for two gens, then switch:

| Stage | Start | What the net must learn |
|-------|--------|-------------------------|
| 1 pad slam | ~250 m, falling ~15 m/s, light wind, pitch plane | Time one latched burn, touch the pad |
| 2 2 km slam | ~2 km, 40–120 m/s down, ±200 m east | Ignition timing |
| 3 6DOF | Same energy, small tilt/rate | Both gimbals + roll fin |
| 4 wind | 6DOF, unobserved sampled wind | Infer gusts from velocity drift |
| 5 glide | ~20 km, ~380 m/s | Fins, then the same slam |
| 6 RTLS | ~80 km, ~2 km/s | Entry, glide, and landing |

The green dashed line in the scene is a human-only slam/divert tube (`v* = √(2 a_eff h)` toward the pad). It is **not** an MLP input and is **not** in the fitness.

Population 128 is for the swarm view (Hansen’s default λ is ~20 at this dimension).

## Approximate vehicle numbers

| Quantity | Value used | Notes |
|----------|------------|--------|
| Stage diameter | 3.66 m | Public |
| Stage + interstage length | 47 m | Full stack ~70 m |
| Stage-1 dry mass | 25 600 kg | Published estimates ~22–27 t |
| Merlin 1D SL / vac | 845 / 914 kN | Block 5 public figures |
| Merlin 1D Isp SL / vac | 282 / 311 s | Same |
| Landing / entry engines | 1 or 3 | Center or cluster |
| Grid fins | 4 × ~1.8 m² lattices | Titanium Block 5; photos / patents |
| Pad | LZ-1, 28.4856°N 80.5444°W | Public coordinates |

## Architecture

```
crates/sim     Rust physics + CMA-NeuroES  →  wasm32-unknown-unknown
web/           Vite + TypeScript + Three.js
```

## Controls

- **Start training / Pause / Reset episode**
- **Reset**: wipe the saved CMA brain in this browser and restart pad slam (hard-reload does not)
- **Reset latest phase**: undo the current stage using the last promote snapshot; drops back one gate if there is no snapshot
- **Autopilot demo**: fly the scripted tracker on a full RTLS start
- **Destruction / Storm / Shear / Wind** (storm/shear pin; wind is nominal intensity)
- **Cameras**: chase, pad, orbital
- **Time warp**: 1 / 5 / 25 / 100 / 250×
