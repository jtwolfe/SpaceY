# SpaceY

Browser simulation of a **Falcon 9-class first stage** coming in from near-vacuum through a grid-fin atmospheric glide to a propulsive landing — and an in-browser **CMA-ES** trainer for a residual control policy.

Two selectable scenarios:

| Scenario | Start | Notes |
|----------|--------|--------|
| **RTLS** (default) | ~80 km, ~2.0–2.1 km/s Earth-relative | First-stage reentry after boostback. This is v1. |
| **LEO deorbit** | ~220 km, **~7.8 km/s inertial** (~7.3–7.5 km/s ground), near-vacuum | Same vehicle from circular LEO-class energy: retrograde deorbit, hypersonic entry, grid-fin glide, propulsive landing. First stages do not actually reach orbit; this is the energy-class gap vs RTLS. |

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
| `npm test`     | Native Rust unit tests (atmosphere, frames, episode termination, LEO start) |
| `npm run build`| Production static build in `web/dist`             |

CI on `main` and pull requests runs `cargo test -p spacey_sim` and a `wasm32-unknown-unknown` release build (fail closed if the crate does not compile).

## What the sim models

- **6DOF rigid body** in ECI, with ECEF / geodetic conversions and a rotating Earth (WGS-84 + J2).
- **US Standard Atmosphere 1976** (NASA-TM-X-74335 / NOAA-S/T 76-1562) from sea level through 86 km, exponential tail above.
- **Aerodynamics**: Mach-dependent drag, AoA lift, grid fins as control surfaces, a combined CP that weathercocks tail-first. Coherent and dimensioned — not a NASA aero table.
- **Propulsion**: up to three Merlin-class engines, throttle 40–100%, gimbal, Isp mix of SL/vac, fuel-mass depletion.
- **Wind / weather**: layered Florida-east-coast caricature plus Ornstein–Uhlenbeck gusts. Optional **storm** (stronger surface flow, larger gusts, +10% density) and **shear** (amplified layer-to-layer speed/direction contrast, slightly thinner mid-atmosphere). Synthetic only — not live METAR.
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
| Scenario start fuel | 40 000 kg RTLS / 78 000 kg LEO | After ascent + boostback; LEO carries more for the hypersonic capture + a pad-theater landing stash (still ≪ 396 t ascent) |
| Merlin 1D SL / vac | 845 / 914 kN | Block 5 public figures |
| Merlin 1D Isp SL / vac | 282 / 311 s | Same |
| Landing / entry engines | 1 or 3 | Center or cluster |
| Grid fins | 4 × ~1.8 m² | Titanium Block 5; photos / patents |
| Pad | LZ-1, 28.4856°N 80.5444°W | Public coordinates |

**RTLS** start is an **approximate first-stage reentry**: ~80 km, ~2.0–2.1 km/s Earth-relative, westbound toward LZ-1.

**LEO** start is a circular ~220 km / ~7.8 km/s inertial state on a plane that overflies LZ-1 after a retrograde deorbit and a half-rev coast. Density at that altitude is thermospheric (~10⁻¹⁰ kg/m³). Use 100–250× time warp for the exoatmospheric coast. The same Merlin-class stack then flies hypersonic entry → grid fins → landing burn. Surviving 7.8 km/s on a first-stage airframe is the training problem — not a claim that Falcon 9 stages do this.

The LEO **nominal** (zero residual, destruction on) is built to **soft-land at LZ-1** a meaningful fraction of the time: vacuum RCS holds tail-first through coast, the pad-ENU corridor is ignored until the landing theater, deorbit is a single-engine ~50 m/s burn that latches, and the entry law is a Q-hold plus an inbound-only capture burn. Periapsis is placed ~724 km west and slightly north of the pad so the ~500 km skip after a 5 km/s / 65 km overflight crosses LZ-1 (a due-east periapsis is the orbit apex and would walk ~27 km south). The Q-hold may spend down to a ~3.8 t stash so the pulse brakes instead of coasting into a 249 kPa spike; the landing burn then commits in the pad theater and suicide-brakes into the success box instead of hovering a T/W>1 stack at 1 km or sliding 50 km east. ECI→ECEF uses −ω t so a zero-ground-speed stack actually stays over LZ-1 (the opposite sign walked ~0.8 km/s east). LEO structural limits are higher than RTLS (Q 250 kPa / 18 g) — still fatal for an unburned 7.8 km/s dive, not a destruction-off cheat.

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
- **Scenario**: RTLS (default) or LEO deorbit
- **Destruction** toggle (default on)
- **Storm / shear** weather toggles (on top of the wind scale)
- **Wind** scale 0–2×
- **Cameras**: chase, pad, orbital overview (orbital camera frames both pad and vehicle at Earth scale)
- **Time warp**: 1 / 5 / 25 / 100 / 250× (needed for the LEO coast)
