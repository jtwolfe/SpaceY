# SpaceY

Browser gym for a **Falcon 9-class first stage**: 6DOF physics, grid-fin aero, cold-gas RCS, Merlin 1/3, and an in-browser **CMA-NeuroES** trainer.

The mission is **RTLS**: recover the booster to LZ-1. Training starts as a 2D pad slam and auto-promotes through 2 km, 6DOF, wind, glide, then the full ~80 km RTLS start. The net sees only pad-frame pose, velocity, rates, and fuel, and writes throttle, gimbal, and grid fins.

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
  - **Merlins**: 1 or 3 engines, 40–100% throttle, 2-axis gimbal, Isp mix SL/vac. Once lit, a burn lasts ≥2.5 s; a restart waits ≥6 s. 10 Hz on/off is a relight, not a throttle.
  - **Grid fins**: four lattices at the interstage, X-mixer, axial drag even at δ=0, weathercock from `r × F` (not a lumped CP hack).
  - **RCS**: cold-gas inner loop for the **autopilot demo** only. Training (Policy) has RCS off so the net owns attitude.
- **Wind / weather**: Florida-east-coast caricature + OU gusts. Optional storm / shear.
- **Destruction** (default on): max-Q, over-G, q-alpha, AoA, spin, hard impact.
- **Success**: engine ~12 m, ≲8 m/s, ≲4 m/s horizontal, within 20 m of the pad, ≲12° tilt, intact. A toy-soft landing, not a 26 m/s slap.

## How training works

The **policy is the controller**. A small tanh MLP (14 → 8 hidden → 6 actions, 174 weights) reads pad-ENU engine position, velocity, body→ENU quaternion, body rate, and fuel. It writes throttle (off or 40–100%), two-axis TVC, and three fin axes. Actions are held at **10 Hz**. The plant latches ignition (min burn / restart delay); extra relights cost fitness. One landing engine. No inner attitude PD, no wind/q/Mach/phase inputs, no reference trajectory in the observation.

Weights are evolved with **CMA-ES** (CMA-NeuroES). No backprop. If fitness plateaus, one hidden unit is added (CMA-TWEANN style, new weights start at 0, covariance reset). Cap is 24 hidden so the browser covariance stays small.

Zero weights keep engines **off**. CMA has to learn to light a Merlin. Promoting a stage **keeps** the champion weights and resets covariance.

**Autopilot demo** is the old hand-written tracker on a full RTLS start. It is not in the training loop.

Curriculum is internal. After three generations at ≥30% true lands, the trainer advances one stage:

| Stage | Start | What the net must learn |
|-------|--------|-------------------------|
| 1 pad slam | ~80 m, nearly still, pitch plane | Light late, stay upright |
| 2 2 km slam | ~2 km, 40–120 m/s down, ±200 m east | Ignition timing |
| 3 6DOF | Same energy, full attitude | Both gimbals + roll fin |
| 4 wind | 6DOF, unobserved ~1× wind | Infer gusts from velocity drift |
| 5 glide | ~20 km, ~380 m/s | Fins, then the same slam |
| 6 RTLS | ~80 km, ~2 km/s | Entry, glide, and landing |

The green dashed line in the scene is a human-only slam/divert tube (`v* = √(2 a_eff h)` toward the pad). It is **not** an MLP input. Fitness adds a dense tracking cost plus the existing land jackpot.

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
- **Autopilot demo**: fly the scripted tracker on a full RTLS start
- **Destruction / Storm / Shear / Wind**
- **Cameras**: chase, pad, orbital
- **Time warp**: 1 / 5 / 25 / 100 / 250×
