import { useSpacey } from "@/game/store";
import { missionLabel } from "@/game/scenario";

function fmt(n: number, d = 0) {
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(d);
}

export function Hud() {
  const snap = useSpacey((s) => s.snap);
  const brain = useSpacey((s) => s.brain);
  const pilot = useSpacey((s) => s.pilot);
  const warp = useSpacey((s) => s.warp);
  const genNote = useSpacey((s) => s.genNote);
  if (!snap) return null;
  const phase = snap.phase.toUpperCase();
  const term = snap.term !== "none" ? snap.term : phase;
  const ok = snap.term === "landed";
  const bad = snap.term === "destroyed" || snap.term === "miss" || snap.term === "fuel";
  return (
    <div className="pointer-events-none absolute inset-x-0 top-0 z-10 flex flex-col gap-3 p-4 pt-[max(1rem,env(safe-area-inset-top))] md:flex-row md:justify-between">
      <div className="rounded-lg bg-bg-elevated/80 px-4 py-3 ring-1 ring-border backdrop-blur-sm">
        <p className="font-mono text-xs tracking-widest text-muted">
          SPACEY · {missionLabel(snap.energy)}
          {snap.watch ? ` · WATCH ${warp}×  GEN ${snap.watchGen ?? 0} #${snap.watchIdx ?? 0}` : ""}
        </p>
        <p
          className={`mt-1 font-mono text-lg font-medium tabular-nums ${ok ? "text-ok" : bad ? "text-bad" : "text-steel"}`}
        >
          {term}
        </p>
        <p className="mt-2 grid grid-cols-2 gap-x-6 gap-y-1 font-mono text-xs tabular-nums text-fg">
          <span>ALT {fmt(snap.engineAlt, 0)} m</span>
          <span>SPD {fmt(snap.speed, 1)} m/s</span>
          <span>RNG {fmt(snap.range, 0)} m</span>
          <span>TILT {fmt(snap.tiltDeg, 1)}°</span>
          <span>FUEL {fmt(snap.fuel, 0)} kg</span>
          <span>
            MER {snap.nEngines} · {fmt(snap.throttle * 100, 0)}%
          </span>
          <span>Q {fmt(snap.qkpa, 1)} kPa</span>
          <span>T+{fmt(snap.t, 1)} s</span>
        </p>
        <p className="mt-2 font-mono text-[10px] text-muted">Drag to orbit · scroll zoom · Chase / Pad / Orbit</p>
      </div>
      {pilot !== "train" ? (
        <div className="rounded-lg bg-bg-elevated/80 px-4 py-3 ring-1 ring-border backdrop-blur-sm md:text-right">
          <p className="font-mono text-xs tracking-widest text-muted">
            {pilot === "manual" ? "MANUAL" : "AUTOPILOT"}
          </p>
          <p className="mt-1 font-mono text-sm tabular-nums text-steel">
            gen {brain.gen} · E {brain.energy.toFixed(2)} · land {(brain.landRate * 100).toFixed(0)}%
          </p>
          {genNote ? <p className="mt-1 max-w-xs font-mono text-xs text-muted md:ml-auto">{genNote}</p> : null}
        </div>
      ) : (
        <div className="hidden md:block md:w-80" />
      )}
    </div>
  );
}
