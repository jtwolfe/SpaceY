import { N_HIDDEN, N_OUT, N_WEIGHTS, TOPOLOGY } from "@/game/policy";
import { GATE_NEED, GATE_RATE, POP } from "@/game/trainer";
import { useSpacey } from "@/game/store";
import { LADDER, missionLabel } from "@/game/scenario";

function barTone(term: string, landed: boolean, hero: boolean) {
  if (landed) return "bg-ok";
  if (term === "fly") return hero ? "bg-steel" : "bg-steel/55";
  if (term === "destroyed") return "bg-bad";
  return "bg-bad/70";
}

export function GymPanel() {
  const gym = useSpacey((s) => s.gym);
  const pilot = useSpacey((s) => s.pilot);
  const genNote = useSpacey((s) => s.genNote);
  const warp = useSpacey((s) => s.warp);
  if (pilot !== "train") return null;
  const hidden = gym?.hidden ?? Array.from({ length: N_HIDDEN * 2 }, () => 0);
  const h1 = hidden.slice(0, N_HIDDEN);
  const h2 = hidden.slice(N_HIDDEN, N_HIDDEN * 2);
  const outputs = gym?.outputs ?? Array.from({ length: N_OUT }, () => 0);
  const pop = gym?.pop ?? [];
  const outLabels = ["thr", "gimY", "gimZ", "finP", "finY", "finR", "n", "rcsY", "rcsZ", "finR2"];
  const nLive = gym?.nLive ?? pop.filter((p) => p.alive).length;
  const nLand = gym?.nLand ?? 0;
  const nDead = gym?.nDead ?? 0;
  const stage = gym?.stage ?? "pad";
  const stageIdx = Math.max(0, LADDER.findIndex((r) => r.id === stage));
  const gateStreak = gym?.gateStreak ?? 0;
  const atTop = stageIdx >= LADDER.length - 1;
  return (
    <div className="pointer-events-none absolute right-0 top-0 z-10 flex w-[min(17rem,calc(100%-0.75rem))] flex-col gap-2 p-3 pt-[max(0.75rem,env(safe-area-inset-top))] md:w-80 md:p-4">
      <div className="rounded-lg bg-bg-elevated/85 px-4 py-3 ring-1 ring-border backdrop-blur-sm">
        <p className="font-mono text-xs tracking-widest text-muted">CMA-ES · IN THIS TAB</p>
        <p className="mt-1 font-mono text-sm text-steel">
          {TOPOLOGY} tanh · {N_WEIGHTS} weights · λ {POP}
        </p>
        <p className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 font-mono text-xs tabular-nums text-fg">
          <span>GEN {gym?.gen ?? 0}</span>
          <span>LAND {((gym?.landRate ?? 0) * 100).toFixed(0)}%</span>
          <span>σ {(gym?.sigma ?? 0).toFixed(2)}</span>
          <span>EP {gym?.episodes ?? 0}</span>
          <span>|w| {(gym?.weightNorm ?? 0).toFixed(2)}</span>
          <span>{missionLabel(gym?.energy ?? 0)}</span>
        </p>
        <p className="mt-2 font-mono text-[11px] tabular-nums text-steel">
          {nLive} flying · {nLand} landed · {nDead} out · gym t+{(gym?.simT ?? 0).toFixed(1)}s
        </p>
        <p className="mt-2 font-mono text-[11px] text-steel">
          {atTop
            ? "RTLS unlocked · holding the top rung"
            : `Gate ${gateStreak}/${GATE_NEED} gens at ≥${Math.round(GATE_RATE * 100)}% land → ${LADDER[stageIdx + 1]?.label}`}
        </p>
        <div className="mt-2 flex gap-1">
          {LADDER.map((r, i) => (
            <div
              key={r.id}
              className={`h-1.5 flex-1 rounded-sm ${
                i < stageIdx ? "bg-ok/80" : i === stageIdx ? "bg-steel" : "bg-border"
              }`}
            />
          ))}
        </div>
        <p className="mt-2 hidden font-mono text-[11px] leading-relaxed text-muted md:block">
          Gym trains at full speed. Camera follows one genome at {warp}×; after a miss it respawns from the live
          generation. Stages only move forward.
        </p>
        {genNote ? <p className="mt-2 font-mono text-[11px] text-ok">{genNote}</p> : null}
      </div>
      <div className="rounded-lg bg-bg-elevated/85 px-4 py-3 ring-1 ring-border backdrop-blur-sm">
        <p className="font-mono text-xs tracking-widest text-muted">POPULATION · 12 PARALLEL</p>
        <div className="mt-2 flex items-end gap-1" style={{ height: 56 }}>
          {(pop.length ? pop : Array.from({ length: POP }, () => null)).map((p, i) => {
            const fit = p && Number.isFinite(p.fit) ? p.fit : 0;
            const h = p?.alive ? Math.min(1, Math.max(0.2, (fit + 8000) / 20000)) : Math.min(1, Math.max(0.08, (fit + 4000) / 16000));
            return (
              <div
                key={i}
                className={`min-h-1 flex-1 rounded-sm ${p ? barTone(p.term, p.landed, p.hero) : "bg-border"} ${p?.hero ? "ring-1 ring-fg" : ""}`}
                style={{ height: `${h * 100}%` }}
                title={p ? `#${i} ${p.term} r=${p.range.toFixed(0)}` : ""}
              />
            );
          })}
        </div>
        <p className="mt-2 font-mono text-[10px] text-muted">green land · steel flying · red miss · ring = watch rocket</p>
      </div>
      <div className="hidden rounded-lg bg-bg-elevated/85 px-4 py-3 ring-1 ring-border backdrop-blur-sm md:block">
        <p className="font-mono text-xs tracking-widest text-muted">WATCH H1 tanh</p>
        <HiddenRow values={h1.length ? h1 : Array.from({ length: N_HIDDEN }, () => 0)} />
        <p className="mt-3 font-mono text-xs tracking-widest text-muted">WATCH H2 tanh</p>
        <HiddenRow values={h2.length ? h2 : Array.from({ length: N_HIDDEN }, () => 0)} />
        <p className="mt-3 font-mono text-xs tracking-widest text-muted">
          WATCH RESIDUAL · |y| {(gym?.meanAbsY ?? 0).toFixed(2)}
        </p>
        <div className="mt-2 grid grid-cols-5 gap-1">
          {outputs.map((v, i) => (
            <div key={i} className="flex flex-col items-center gap-1">
              <div className="h-8 w-full overflow-hidden rounded-sm bg-bg-subtle">
                <div
                  className={`w-full ${v >= 0 ? "bg-ok/80" : "bg-bad/80"}`}
                  style={{ height: `${Math.min(100, Math.abs(v) * 100)}%`, marginTop: v >= 0 ? `${100 - Math.abs(v) * 100}%` : 0 }}
                />
              </div>
              <span className="font-mono text-[9px] text-muted">{outLabels[i]}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function HiddenRow({ values }: { values: number[] }) {
  return (
    <div className="mt-2 flex items-end gap-1" style={{ height: 36 }}>
      {values.map((v, i) => (
        <div
          key={i}
          className="flex-1 rounded-sm bg-steel/80"
          style={{ height: `${Math.max(10, Math.abs(v) * 100)}%`, opacity: 0.3 + Math.abs(v) * 0.7 }}
        />
      ))}
    </div>
  );
}
