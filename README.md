# SpaceY

Browser simulation of a **Falcon 9-class first stage** coming in from near-vacuum through a grid-fin atmospheric glide to a propulsive landing — and an in-browser **CMA-ES** trainer for a residual control policy.

Physics and training run entirely in **Rust compiled to WASM**. The scene is **TypeScript + Three.js**. This is a normal `cargo` + `wasm-bindgen` + `vite` toolchain.

**Grok Build, the `grok` CLI, grok TUI, and any Grok Build / Grok coding-agent workflow are not part of this project.**

Numbers below are **approximate public figures**, not live telemetry and not a proprietary aero database.

## Run (one command after the usual Rust + Node install)

Prerequisites: Rust (stable) with `wasm32-unknown-unknown`, [`wasm-bindgen-cli`](https://crates.io/crates/wasm-bindgen) **0.2.100**, and Node.js 20+.

```bash
cargo install wasm-bindgen-cli --version 0.2.100 --locked
rustup target add wasm32-unknown-unknown
npm start
```

That builds the WASM crate and serves the app at [http://localhost:5173](http://localhost:5173).

Other scripts:

| Command        | What it does                                      |
|----------------|---------------------------------------------------|
| `npm test`     | Native Rust unit tests (atmosphere, frames, episode termination) |
| `npm run build`| Production static build in `web/dist`             |

## What the sim models

- **6DOF rigid body** in ECI, with ECEF / geodetic conversions and a rotating Earth (WGS-84 + J2).
- **US Standard Atmosphere 1976** (NASA-TM-X-74335 / NOAA-S/T 76-1562) from sea level through 86 km, exponential tail above.
- **Aerodynamics**: Mach-dependent drag, AoA lift, grid fins as control surfaces, a combined CP that weathercocks tail-first. Coherent and dimensioned — not a NASA aero table.
- **Propulsion**: up to three Merlin-class engines, throttle 40–100%, gimbal, Isp mix of SL/vac, fuel-mass depletion.
- **Wind**: layered Florida-east-coast caricature plus Ornstein–Uhlenbeck gusts (not a live weather product).
- **Destruction** (default **on**): max-Q, over-G, q-alpha, excessive AoA, spin, hard impact. Toggle in the UI.
- **Fuel-to-land bound**: conservative analytic check once in the lower atmosphere / landing burn.
- **Corridor**: ground-track crossrange from the start→pad line; tightens with altitude.
- **Success**: upright, low residual speed, engines near LZ-1, vehicle intact.

## Approximate vehicle numbers

Cited from SpaceX Falcon 9 user's guide figures, FAA/environmental filings, and commonly quoted secondary summaries of those documents. Treat as order-of-magnitude.

| Quantity | Value used | Notes |
|----------|------------|--------|
| Stage diameter | 3.66 m | Public |
| Stage + interstage length | 47 m | Full stack ~70 m |
| Stage-1 dry mass | 25 600 kg | Published estimates ~22–27 t |
| Ascent propellant capacity | 395 700 kg | LOX + RP-1; **not** the start load |
| Scenario start fuel | 40 000 kg | After ascent + boostback (RTLS-class remainder) |
| Merlin 1D SL / vac | 845 / 914 kN | Block 5 public figures |
| Merlin 1D Isp SL / vac | 282 / 311 s | Same |
| Landing / entry engines | 1 or 3 | Center or cluster |
| Grid fins | 4 × ~1.8 m² | Titanium Block 5; photos / patents |
| Pad | LZ-1, 28.4856°N 80.5444°W | Public coordinates |

Start state is an **approximate first-stage reentry**: ~80 km, ~2.0–2.1 km/s Earth-relative, westbound toward LZ-1. First stages do not reach orbital velocity; this is reentry-class, not a 7.8 km/s LEO deorbit.

## How training works

A **nominal tracker** (entry-burn energy management, tail-first weathercock / grid fins, hover-slam landing) is always running.

On top of that, a **linear residual policy** (16 features → 6 actions: throttle, gimbal, fins; 102 weights) is trained with **CMA-ES** (rank-μ + rank-1 covariance, Cholesky sampling). No backprop. Population fitness is drawn in the HUD each generation.

The display vehicle flies the current champion. Training evaluates a population of 12 in WASM between animation frames.

Success score is large and positive; corridor / fuel / breakup / timeout are large negatives. CMA-ES is a later-PPO stand-in: it fits continuous control and parallelizes in the browser.

## Architecture

```
crates/sim     Rust physics + CMA-ES  →  wasm32-unknown-unknown
web/           Vite + TypeScript + Three.js
```

Bevy-on-WASM was considered and skipped: longer compile, heavier download, no advantage for this scene.

## Controls

- **Start training / Pause / Reset episode**
- **Destruction** toggle (default on)
- **Wind** scale 0–2×
- **Cameras**: chase, pad, orbital overview
- **Time warp**: 1 / 5 / 25 / 100× (useful in the exoatmospheric coast)
