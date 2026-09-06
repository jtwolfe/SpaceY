import { CircleDot, Dna, Rocket } from "lucide-react";
import { useSpacey } from "@/game/store";
import { N_HIDDEN_LAYERS_MAX, N_WEIGHTS, TOPOLOGY, nWeights } from "@/game/policy";
import { MU, POP } from "@/game/trainer";

export function StartOverlay() {
  const started = useSpacey((s) => s.started);
  const start = useSpacey((s) => s.start);
  if (started) return null;
  return (
    <div className="pointer-events-none absolute inset-0 z-20 flex items-end justify-center p-4 pb-[max(1.25rem,env(safe-area-inset-bottom))]">
      <div className="pointer-events-auto w-full max-w-md rounded-lg bg-bg-elevated/92 p-5 ring-1 ring-border backdrop-blur-sm md:p-6">
        <p className="font-mono text-xs tracking-[0.22em] text-muted">LZ-1 · BLOCK 5 CLASS · LIVE GYM</p>
        <h1 className="mt-2 text-3xl font-medium tracking-tight text-fg md:text-4xl">SpaceY</h1>
        <p className="mt-3 text-pretty text-sm leading-relaxed text-muted">
          Residual net starts at {TOPOLOGY} ({N_WEIGHTS} w) and grows a hidden layer as CMA unlocks each rung — up to{" "}
          {N_HIDDEN_LAYERS_MAX} hidden ({nWeights(N_HIDDEN_LAYERS_MAX)} w). Hansen sep-CMA-ES (λ {POP}, μ {MU}) updates the
          mean, step-size σ, and diagonal covariance in this tab. Gated: Pad → 2 km → Glide → RTLS. The residual sees a
          receding-horizon goal from each booster’s live state — when to light, remaining divert, when to stand up — not
          a frozen spawn path. 2 km hops start inbound. Unpowered GNC holds near-vertical with engines toward the pad so
          grid fins can kill miss; Merlin waits for suicide altitude and lands on one engine — three-wide is only a
          short high-throttle pulse when one Merlin cannot stop, never three at 40%. Once the landing burn is lit it
          stays lit to gear height; a loft drops to min throttle instead of dumping the cluster. The score pays
          ballistic miss killed before light and taxes engine-on time, three-wide (especially at the floor), and
          thrusting away while still short — no bonus for pointing the nose at the pad.
        </p>
        <p className="mt-3 font-mono text-xs text-steel">Drag to look · scroll zoom · Chase / Pad / Orbit</p>
        <div className="mt-6 flex flex-col gap-2">
          <button
            type="button"
            onClick={() => start("train")}
            className="inline-flex min-h-11 w-full items-center justify-center gap-2 rounded-md bg-steel px-4 font-medium text-bg transition-transform duration-150 hover:opacity-90 active:scale-[0.98]"
          >
            <Dna className="size-4" />
            Watch the gym
          </button>
          <button
            type="button"
            onClick={() => start("autopilot", "slam")}
            className="inline-flex min-h-11 w-full items-center justify-center gap-2 rounded-md bg-bg-subtle px-4 font-medium text-steel ring-1 ring-border transition-transform duration-150 hover:bg-bg active:scale-[0.98]"
          >
            <CircleDot className="size-4" />
            Watch a 2 km hop
          </button>
          <button
            type="button"
            onClick={() => start("autopilot")}
            className="inline-flex min-h-11 w-full items-center justify-center gap-2 rounded-md bg-bg-subtle px-4 font-medium text-steel ring-1 ring-border transition-transform duration-150 hover:bg-bg active:scale-[0.98]"
          >
            <Rocket className="size-4" />
            Watch a GNC pad landing
          </button>
        </div>
      </div>
    </div>
  );
}
