# SpaceY — in-tab CMA gym

TypeScript rewrite of the Falcon 9–class RTLS trainer. A 14→8→8→10 tanh residual net (282 weights) is evolved in the browser with CMA-ES (λ=12). The gym trains at full speed; the camera follows one rocket at watch speed (1× / 4× / 16×). After a miss, the next watch flight is sampled from the live generation.

Training is a gated ladder: **Pad → 2 km → Glide → RTLS**. Two consecutive generations at ≥40% land unlock the next rung. No auto-drop.

A legal land is a 6k floor plus a 6k squared bullseye that collapses toward the 20 m rim, with an extra quadratic range tax. Touching down still dwarfs every miss; landing on the mark outranks a rim land by thousands of points.

This branch is the Grok in-browser gym (not the Rust/WASM tree on `main` / `rework/full-rtls`).

```
npm install
npm run dev
```
