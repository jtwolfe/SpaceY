import type { ReactNode } from "react";
import { Camera, Gauge, Pause, Play, RotateCcw, Rocket, Dna } from "lucide-react";
import { LADDER } from "@/game/scenario";
import { useSpacey, type AppPilot, type CamMode } from "@/game/store";

const pilots: { id: AppPilot; label: string }[] = [
  { id: "train", label: "Train" },
  { id: "autopilot", label: "Autopilot" },
  { id: "manual", label: "Manual" },
];

const cams: { id: CamMode; label: string }[] = [
  { id: "chase", label: "Chase" },
  { id: "pad", label: "Pad" },
  { id: "orbit", label: "Orbit" },
];

export function Dock() {
  const mission = useSpacey((s) => s.mission);
  const pilot = useSpacey((s) => s.pilot);
  const cam = useSpacey((s) => s.cam);
  const warp = useSpacey((s) => s.warp);
  const paused = useSpacey((s) => s.paused);
  const snap = useSpacey((s) => s.snap);
  const setMission = useSpacey((s) => s.setMission);
  const setPilot = useSpacey((s) => s.setPilot);
  const setCam = useSpacey((s) => s.setCam);
  const cycleWarp = useSpacey((s) => s.cycleWarp);
  const setPaused = useSpacey((s) => s.setPaused);
  const bumpSeed = useSpacey((s) => s.bumpSeed);
  const resetBrain = useSpacey((s) => s.resetBrain);
  const train = pilot === "train";
  const currentIdx = LADDER.findIndex((m) => m.id === mission);

  return (
    <div className="absolute inset-x-0 bottom-0 z-10 p-3 pb-[max(0.75rem,env(safe-area-inset-bottom))]">
      <div className="mx-auto flex max-w-5xl flex-col gap-2 rounded-lg bg-bg-elevated/90 p-2 ring-1 ring-border backdrop-blur-sm md:flex-row md:items-center md:justify-between">
        <div className="flex flex-wrap items-center gap-1">
          {train ? <span className="px-2 font-mono text-[10px] tracking-widest text-muted">GATED</span> : null}
          {LADDER.map((m, i) => {
            const now = mission === m.id;
            const done = train && i < currentIdx;
            const locked = train && i > currentIdx;
            return (
              <Chip
                key={m.id}
                active={now}
                done={done}
                locked={locked}
                onClick={train ? undefined : () => setMission(m.id)}
              >
                {m.label}
              </Chip>
            );
          })}
        </div>
        <div className="flex flex-wrap gap-1">
          {pilots.map((p) => (
            <Chip key={p.id} active={pilot === p.id} onClick={() => setPilot(p.id)}>
              {p.id === "train" ? (
                <Dna className="size-3.5" />
              ) : p.id === "manual" ? (
                <Gauge className="size-3.5" />
              ) : (
                <Rocket className="size-3.5" />
              )}
              {p.label}
            </Chip>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-1">
          {cams.map((c) => (
            <Chip key={c.id} active={cam === c.id} onClick={() => setCam(c.id)}>
              <Camera className="size-3.5" />
              {c.label}
            </Chip>
          ))}
          <Chip active={false} onClick={cycleWarp}>
            {train ? `Watch ${warp}×` : `${warp}×`}
          </Chip>
          <Chip active={paused} onClick={() => setPaused(!paused)}>
            {paused ? <Play className="size-3.5" /> : <Pause className="size-3.5" />}
            {paused ? "Resume" : "Pause"}
          </Chip>
          <Chip active={false} onClick={() => bumpSeed()}>
            <RotateCcw className="size-3.5" />
            Relight
          </Chip>
          <Chip active={false} onClick={() => resetBrain()}>
            Reset net
          </Chip>
        </div>
      </div>
      {snap && snap.term !== "none" && !train ? (
        <p className="mx-auto mt-2 max-w-4xl text-center font-mono text-xs text-muted">
          {snap.term === "landed"
            ? "Legs down. Relight, or open Train to watch CMA-ES."
            : "GNC missed the land box. Train lets twelve residual nets search."}
        </p>
      ) : null}
    </div>
  );
}

function Chip({
  active,
  onClick,
  children,
  done = false,
  locked = false,
}: {
  active: boolean;
  onClick?: () => void;
  children: ReactNode;
  done?: boolean;
  locked?: boolean;
}) {
  const tone = locked
    ? "cursor-default bg-bg-subtle text-muted ring-1 ring-border"
    : active
      ? "bg-steel text-bg"
      : done
        ? "cursor-default bg-ok/20 text-ok ring-1 ring-ok/40"
        : "bg-bg-subtle text-steel ring-1 ring-border hover:bg-bg";
  return (
    <button
      type="button"
      onClick={locked ? undefined : onClick}
      disabled={locked || !onClick}
      className={`inline-flex min-h-11 items-center gap-1.5 rounded-sm px-3 font-mono text-xs tracking-wide disabled:opacity-100 ${tone}`}
    >
      {children}
    </button>
  );
}
