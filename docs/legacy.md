# Legacy code

SpaceY started as an experiment: run a landing-policy search **in the browser**, first as Rust compiled to WASM, then as this TypeScript gym. The earlier work is still in the tree so the experiment stays inspectable. It is not maintained.

## `legacy/rust`

Snapshot of the old default branch (`bcab81b`, previously `main`).

What it was:

- Rust 6DOF (ECI, WGS-84, atmosphere, Merlin 1/3, grid fins, RCS) compiled to WASM
- A TypeScript + Three.js scene under `web/`
- CMA-NeuroES with a different observation, a different curriculum (pad → 2 km → 6DOF → wind → glide → RTLS), and a policy that *was* the controller rather than a residual on GNC

What it is now:

- **Reference only.** Not imported by the app. Not in the Vite graph. Not expected to `npm start` or `cargo test` cleanly on a current toolchain
- Save format, land box, and scoring are **not** compatible with `src/game/`
- Bits are stale or broken; that is why they were replaced, not deleted

If you want the git-native copy instead of this snapshot, `bcab81b` and branches such as `rework/full-rtls` are still on the remote.

## What to read instead

The live system is [`src/game/`](../src/game/). Start at [`docs/how-it-works.md`](how-it-works.md).
