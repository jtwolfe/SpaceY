# How the current gym works

SpaceY is a single-tab trainer. The browser owns the physics, the autopilot, the residual net, and the CMA-ES loop. There is no training server.

## Loop

1. **Splash.** The overlay is a briefing. The gym is paused until **Start training**.
2. **Watch vs gym.** CMA evaluates a population of 64 boosters as fast as the tab will go. The camera follows one live genome at 1× / 2× / 4× / 8× / 16×. After that rocket terms, the next watch body is sampled from the current generation.
3. **Save.** The mean, σ, diagonal C, layer count, and gated energy are written to `localStorage`. Reload continues; **Reset net** zeros it.

## Ladder

| Rung | Energy | Rough start |
|------|--------|-------------|
| Pad | 0 | ~250 m hover-slam onto LZ-1 |
| 2 km | 0.22 | Inbound hop, suicide light |
| Glide | 0.68 | ~20 km up, ~11 km east, grid fins then slam |
| RTLS | 1 | ~80 km / ~2 km/s entry, then the same landing |

Unlock is two consecutive generations at ≥40% land. The gate never drops a rung on its own. Each unlock (and some σ plateaus) may splice in another identity residual block.

## Split of labour

- **`guidance.ts` / `sim.ts`** — phases (exo, entry, glide, landing), 3-Merlin entry, 1-Merlin suicide, grid-fin fade with dynamic pressure.
- **`policy.ts`** — growable residual MLP. Forward at zero weights is GNC. On RTLS the residual is frozen through entry / high glide so old glide weights cannot tumble the stack.
- **`trainer.ts`** — sep-CMA-ES (λ 64, μ 32, CSA σ, diagonal C). Parents on RTLS are landers and near-pad misses, not high-scoring far fly-bys.
- **`scene.ts` / `world.ts`** — WGS84 Earth, LZ-1 / LZ-2, chase / pad / orbit cameras.

Autopilot and Manual in the dock fly the same plant without updating CMA. They are demos, not a second trainer.

## What “good” looks like

A legal land is gear on the inner pad disk, slow and upright enough to count. Fitness still prefers the mark over a rim touch, and prefers killing miss while unpowered over a last-second TVC hook. Residual capacity (H) is allowed to run long on RTLS; extra identity layers only help if CMA uses them.
