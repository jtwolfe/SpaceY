# SpaceY — in-tab CMA gym

TypeScript rewrite of the Falcon 9–class RTLS trainer. A 210-weight residual net is evolved in the browser with CMA-ES (λ=12). The gym trains at full speed; the camera follows one rocket at 1×. After a miss, the next 1× flight is sampled from the live generation.

This branch is the Grok in-browser gym (not the Rust/WASM tree on `main` / `rework/full-rtls`).

```
npm install
npm run dev
```
