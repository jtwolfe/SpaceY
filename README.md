# SpaceY

A **local, in-browser CMA-ES gym** for a Falcon 9–class first stage **return to landing site**.

The plant, the GNC, and the trainer all run in this tab. Nothing is uploaded. A population of boosters trains at full speed; the camera follows one genome at 1–16×. Hit **Start training** and sep-CMA-ES walks a residual net through Pad → 2 km → Glide → full RTLS.

This whole repo is an experiment in **ML and browser-side training**: can a small evolutionary strategy, running entirely on the client, climb a real-shaped landing ladder. The live app is TypeScript. Earlier Rust/WASM trainers are frozen under [`legacy/`](legacy/) for reference — old, often broken, not wired into the app.

## Run

```bash
npm install
npm run dev
```

The briefing overlay explains the gym. Training does not start until you click **Start training**. Weights and CMA state live in `localStorage`, so a reload keeps the session.

| Command | What |
|---------|------|
| `npm run typecheck` | `tsc --noEmit` |
| `npx jiti src/game/headless-check.ts` | Headless GNC / trainer sanity |

## How it works

**Ladder.** Four gated rungs: pad hover-slam, 2 km hop, unpowered grid-fin glide, then RTLS from ~80 km. Two consecutive generations at ≥40% land unlock the next rung. No auto-drop.

**Plant + GNC.** TypeScript 6DOF on a WGS84 Earth, LZ-1 / LZ-2 at the Cape. Hand-written guidance flies entry, tail-first glide, and a 1-Merlin suicide. Three-wide is only a short pulse when one engine cannot stop.

**Residual net.** Starts at `21→8→10` (266 weights) and grows identity hidden blocks as rungs unlock or σ plateaus, up to 10 layers. Zero weights mean “GNC only.” The net nudges gimbal, fins, throttle, and cluster from a receding-horizon goal (when to light, remaining divert, when to stand up). It cannot relight a shut engine.

**Trainer.** Hansen sep-CMA-ES in the tab: λ = 64, μ = 32, diagonal covariance, CSA step-size. The gym steps as fast as the browser will go; watch speed is only the camera rocket. Score pays a land jackpot and ballistic miss killed before light, and taxes extra lights, engine-on time, three-wide, and thrusting away while still short of the pad.

More detail: [docs/how-it-works.md](docs/how-it-works.md).

## Layout

| Path | What |
|------|------|
| [`src/game/`](src/game/) | Physics, GNC, residual policy, CMA trainer, three.js scene |
| [`src/components/`](src/components/) | Start overlay, HUD, gym panel, dock |
| [`legacy/`](legacy/) | Frozen earlier experiments. See [docs/legacy.md](docs/legacy.md) |

After start, the dock still has Autopilot / Manual if you want a look-only hop. That is not the training loop.

## Legacy

[`legacy/rust`](legacy/rust) is the previous gym: Rust 6DOF compiled to WASM, a different CMA-NeuroES loop, and a separate Three.js front end. It is **kept as a record of the experiment**, not as a second product. Do not expect it to build, to match current land rates, or to share a save format with the TypeScript trainer.

The old default-branch tip is still in git (`bcab81b`) and on branches such as `rework/full-rtls`.
