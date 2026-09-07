import { Dna } from "lucide-react";
import { useSpacey } from "@/game/store";
import { N_HIDDEN_LAYERS_MAX, N_WEIGHTS, TOPOLOGY, nWeights } from "@/game/policy";
import { MU, POP } from "@/game/trainer";

export function StartOverlay() {
  const started = useSpacey((s) => s.started);
  const start = useSpacey((s) => s.start);
  if (started) return null;
  return (
    <div className="pointer-events-none absolute inset-0 z-20 flex items-end justify-center p-4 pb-[max(1.25rem,env(safe-area-inset-bottom))]">
      <div className="pointer-events-auto w-full max-w-lg rounded-lg bg-bg-elevated/92 p-5 ring-1 ring-border backdrop-blur-sm md:p-6">
        <p className="font-mono text-xs tracking-[0.22em] text-muted">LOCAL CMA-ES · FALCON 9 CLASS RTLS</p>
        <h1 className="mt-2 text-3xl font-medium tracking-tight text-fg md:text-4xl">SpaceY</h1>
        <p className="mt-3 text-pretty text-sm leading-relaxed text-muted">
          A training gym that lives entirely in this tab — no server, nothing uploaded. Hansen sep-CMA-ES (λ {POP}, μ{" "}
          {MU}) fits a residual net on top of Falcon 9 Block 5–class first-stage GNC and walks it through a return to
          LZ-1: Pad slam → 2 km hop → grid-fin glide → full RTLS from ~80 km.
        </p>
        <p className="mt-3 text-pretty text-sm leading-relaxed text-muted">
          The net starts at {TOPOLOGY} ({N_WEIGHTS} w) and grows a hidden layer as each rung unlocks — up to{" "}
          {N_HIDDEN_LAYERS_MAX} hidden ({nWeights(N_HIDDEN_LAYERS_MAX)} w). It sees a receding-horizon goal from live
          state (when to light, remaining divert, when to stand up), not a frozen spawn path. Unpowered GNC holds
          tail-first with engines toward the pad so grid fins can kill miss; Merlin waits for a 1-engine suicide.
          Three-wide is only a short pulse when one Merlin cannot stop. The score pays ballistic miss killed before
          light and taxes extra lights, engine-on time, and thrusting away while still short.
        </p>
        <p className="mt-3 font-mono text-xs text-steel">Drag to look · scroll zoom · Chase / Pad / Orbit · 1–16×</p>
        <button
          type="button"
          onClick={() => start("train")}
          className="mt-6 inline-flex min-h-11 w-full items-center justify-center gap-2 rounded-md bg-steel px-4 font-medium text-bg transition-transform duration-150 hover:opacity-90 active:scale-[0.98]"
        >
          <Dna className="size-4" />
          Start training
        </button>
      </div>
    </div>
  );
}
