import { Dna, Rocket } from "lucide-react";
import { useSpacey } from "@/game/store";
import { N_WEIGHTS, TOPOLOGY } from "@/game/policy";

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
          A {N_WEIGHTS}-weight {TOPOLOGY} net is evolving in this tab. CMA-ES mutates twelve residual controllers at
          full speed. Training is gated: Pad, then 2 km, Glide, then RTLS — each rung unlocks only after two strong
          generations. The camera follows one rocket; Watch 1×–16× sets its speed.
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
