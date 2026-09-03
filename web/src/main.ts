import init, { Engine } from "./pkg/spacey.js";
import { SceneApp, type CamMode } from "./scene";
import type { Snapshot, TrainInfo, GenerationViz } from "./types";

type WasmEngine = Engine & {
  snapshot?: () => Snapshot;
  train_info?: () => TrainInfo;
  generation_viz?: () => GenerationViz;
  set_autopilot?: (on: boolean) => void;
  autopilot?: () => boolean;
  export_brain?: () => string;
  import_brain?: (json: string) => boolean;
  reset_brain?: () => void;
  reset_latest_phase?: () => void;
};

const BRAIN_KEY = "spacey-brain-v3";

function persistBrain(engine: WasmEngine) {
  try {
    if (typeof engine.export_brain === "function") {
      localStorage.setItem(BRAIN_KEY, engine.export_brain());
    }
  } catch {
    /* quota / private mode */
  }
}

function restoreBrain(engine: WasmEngine) {
  try {
    const raw = localStorage.getItem(BRAIN_KEY);
    if (raw && typeof engine.import_brain === "function") {
      engine.import_brain(raw);
    }
  } catch {
    /* ignore */
  }
}

function clearBrain() {
  try {
    localStorage.removeItem(BRAIN_KEY);
  } catch {
    /* quota / private mode */
  }
}

function fromWasm<T>(v: T | Map<string, unknown>): T {
  if (v instanceof Map) return Object.fromEntries(v) as T;
  return v;
}

function readSnap(engine: WasmEngine): Snapshot {
  if (typeof engine.snapshot === "function") {
    const s = engine.snapshot() as Snapshot | Map<string, unknown>;
    return fromWasm(s) as Snapshot;
  }
  return JSON.parse(engine.snapshot_json()) as Snapshot;
}

function readTrain(engine: WasmEngine): TrainInfo {
  if (typeof engine.train_info === "function") return fromWasm(engine.train_info() as TrainInfo);
  return JSON.parse(engine.train_json()) as TrainInfo;
}

function readViz(engine: WasmEngine): GenerationViz | null {
  if (typeof engine.generation_viz !== "function") return null;
  const v = fromWasm(engine.generation_viz() as GenerationViz);
  if (v.live && v.live instanceof Map) v.live = fromWasm(v.live);
  if (v.prev && v.prev instanceof Map) v.prev = fromWasm(v.prev);
  return v;
}

function nums(v: ArrayLike<number> | undefined | null): number[] {
  if (!v) return [];
  if (Array.isArray(v)) return v.map(Number);
  return Array.from(v as ArrayLike<number>, Number);
}

function fmt(n: number, d = 1): string {
  if (!Number.isFinite(n)) return "—";
  return n.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });
}

function weatherNote(s: Snapshot): string {
  const bits: string[] = [];
  bits.push(`${fmt(s.wind_scale ?? 0, 1)}×`);
  const dir = s.weather_dir_off ?? 0;
  if (Math.abs(dir) >= 0.5) bits.push(`${dir >= 0 ? "+" : ""}${fmt(dir, 0)}°`);
  if (s.weather_storm) bits.push("storm");
  if (s.weather_shear) bits.push("shear");
  return bits.join(" · ");
}

function landBoxNote(s: Snapshot): string {
  if (s.success) return "in box";
  if (!s.terminated) return "—";
  const vh = Math.hypot(s.v_enu?.[0] ?? 0, s.v_enu?.[1] ?? 0);
  const bits: string[] = [];
  if (s.engine_alt >= 12) bits.push(`h ${fmt(s.engine_alt, 0)} m`);
  if (s.speed >= 8) bits.push(`v ${fmt(s.speed, 1)} m/s`);
  if (vh >= 4) bits.push(`vh ${fmt(vh, 1)} m/s`);
  if (s.range_h >= 20) bits.push(`pad ${fmt(s.range_h, 0)} m`);
  if (s.tilt_deg >= 8) bits.push(`tilt ${fmt(s.tilt_deg, 0)}°`);
  return bits.join(" · ") || "—";
}

function camForStage(stageN: number): CamMode {
  return stageN >= 5 ? "orbit" : "pad";
}

function paintPhaseDots(el: HTMLElement, t: TrainInfo) {
  const n = t.stage_count || 6;
  const now = Math.max(1, Math.min(n, t.stage_n || 1));
  const ready = !!t.promote_ready;
  el.innerHTML = Array.from({ length: n }, (_, i) => {
    const k = i + 1;
    const cls = k < now ? "done" : k === now ? (ready ? "now ready" : "now") : "";
    return `<span class="${cls}" title="stage ${k}"></span>`;
  }).join("");
}

function row(dl: Element, rows: [string, string, string?][]) {
  dl.innerHTML = rows
    .map(
      ([k, v, cls]) =>
        `<dt>${k}</dt><dd${cls ? ` class="${cls}"` : ""}>${v}</dd>`,
    )
    .join("");
}

function qClass(q: number): string | undefined {
  if (q > 80_000) return "bad";
  if (q > 40_000) return "warn";
  return undefined;
}

async function main() {
  await init();
  const engine = new Engine() as WasmEngine;
  restoreBrain(engine);
  (window as unknown as { spacey: Engine }).spacey = engine;
  let scene: SceneApp | null = null;
  try {
    scene = new SceneApp(document.querySelector("#view")!);
  } catch (err) {
    console.error("WebGL scene failed; HUD will still run.", err);
  }
  (window as unknown as { spaceyScene: SceneApp | null }).spaceyScene = scene;
  const navMap = document.querySelector<HTMLCanvasElement>("#nav-map");

  const veh = document.querySelector("#veh-hud")!;
  const aero = document.querySelector("#aero-hud")!;
  const guid = document.querySelector("#guid-hud")!;
  const spark = document.querySelector<HTMLCanvasElement>("#fit-spark")!;
  const hist = document.querySelector<HTMLCanvasElement>("#fit-hist")!;
  const meta = document.querySelector("#train-meta")!;
  const dots = document.querySelector<HTMLElement>("#phase-dots")!;
  const status = document.querySelector("#mission-status")!;
  const btnTrain = document.querySelector("#btn-train")!;

  let last = performance.now();
  let snap: Snapshot = readSnap(engine);
  let lastVizStamp = -1;
  let vizCache: GenerationViz | null = null;
  let lastStageN = 0;

  const paintSpark = (t: TrainInfo) => {
    const ctx = spark.getContext("2d");
    if (!ctx) return;
    const w = spark.width;
    const h = spark.height;
    ctx.clearRect(0, 0, w, h);
    const best = nums(t.history_best);
    const mean = nums(t.history_mean);
    if (best.length < 2) {
      ctx.fillStyle = "#7f93a6";
      ctx.font = "10px ui-monospace, monospace";
      ctx.fillText("fitness vs generation", 8, h / 2 + 3);
      return;
    }
    const all = best.concat(mean).filter(Number.isFinite);
    let lo = Math.min(...all);
    let hi = Math.max(...all);
    if (hi - lo < 1) {
      lo -= 50;
      hi += 50;
    }
    const xOf = (i: number, n: number) => (i / Math.max(1, n - 1)) * (w - 2) + 1;
    const yOf = (v: number) => h - 3 - ((v - lo) / (hi - lo)) * (h - 8);
    const stroke = (vals: number[], color: string) => {
      ctx.beginPath();
      vals.forEach((v, i) => {
        const x = xOf(i, vals.length);
        const y = yOf(v);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.4;
      ctx.stroke();
    };
    stroke(mean, "#ffb347");
    stroke(best, "#7ee0ff");
  };

  const paintHist = (fits: number[], terms: string[]) => {
    const ctx = hist.getContext("2d");
    if (!ctx) return;
    const w = hist.width;
    const h = hist.height;
    ctx.clearRect(0, 0, w, h);
    if (!fits.length) return;
    const lo = Math.min(...fits);
    const hi = Math.max(...fits);
    const span = Math.max(1, hi - lo);
    const n = fits.length;
    const bw = Math.max(1, w / n);
    for (let i = 0; i < n; i++) {
      const t = (fits[i] - lo) / span;
      const bh = 3 + t * (h - 5);
      const landed = terms[i] === "landed";
      ctx.fillStyle = landed
        ? i === 0
          ? "#5dffb2"
          : "#2f9d6a"
        : `rgb(${Math.round(220 - 90 * t)},${Math.round(70 + 140 * t)},${Math.round(90 + 130 * t)})`;
      ctx.fillRect(i * bw, h - bh, Math.max(1, bw - 0.3), bh);
    }
  };

  const paint = (s: Snapshot, t: TrainInfo, viz: GenerationViz | null) => {
    const term = s.terminated ? s.term.toUpperCase() : s.phase;
    status.textContent = term;
    status.classList.toggle("good", s.success);
    status.classList.toggle("bad", s.terminated && !s.success);
    btnTrain.classList.toggle("on", t.running);

    const groundKm = s.speed / 1000;
    const finDeg = (s.fins || [0, 0, 0]).map((v) => fmt((Number(v) * 180) / Math.PI, 0));
    row(veh, [
      ["T+ s", fmt(s.t, 1)],
      ["Altitude", `${fmt(s.alt / 1000, 2)} km`],
      ["Speed", `${fmt(s.speed, 0)} m/s`],
      ["Ground", `${fmt(groundKm, 2)} km/s`],
      ["Mach", fmt(s.mach, 2)],
      ["Fuel", `${fmt(s.fuel / 1000, 1)} t`],
      ["Throttle", `${fmt(s.throttle * 100, 0)}% × ${s.n_engines}`],
      ["Relights", `${s.relights ?? 0}  (${s.lights ?? 0} lights)`],
      ["Gimbal", `${fmt(((s.gimbal?.[0] ?? 0) * 180) / Math.PI, 1)}/${fmt(((s.gimbal?.[1] ?? 0) * 180) / Math.PI, 1)}°`],
      ["Tilt", `${fmt(s.tilt_deg, 1)}°`],
      ["Fins", `${finDeg[0]}/${finDeg[1]}/${finDeg[2]}°`],
      ["RCS", fmt(s.rcs ?? 0, 2)],
    ]);
    const rho = s.density;
    const rhoStr =
      rho > 0 && rho < 1e-6 ? rho.toExponential(1) : fmt(rho, 4);
    row(aero, [
      ["Q", `${fmt(s.q_dyn / 1000, 1)} kPa`, qClass(s.q_dyn)],
      ["AoA", `${fmt(s.aoa_deg, 1)}°`, Math.abs(s.aoa_deg) > 25 ? "warn" : undefined],
      ["Load", `${fmt(s.accel_g, 1)} g`],
      ["Heat", fmt(s.heat, 0)],
      ["ρ", `${rhoStr} ×${fmt(s.density_scale, 2)}`],
      ["Gust", `${fmt(s.wind_gust, 1)} m/s`],
    ]);
    const stageN = t.stage_n || 1;
    const stageCount = t.stage_count || 6;
    const stageLabel = t.stage_label || "pad slam";
    const guidRows: [string, string, string?][] = [
      ["Mission", "RTLS"],
      ["Train", `${stageN}/${stageCount} ${stageLabel}`],
      ["Phase", s.phase],
      ["Pilot", s.pilot === "autopilot" ? "autopilot" : "policy"],
      ["Obs", s.plane_lock ? "2D pad-ENU" : "6DOF pad-ENU"],
      ["v* slam", `${fmt(s.v_slam ?? 0, 1)} m/s`],
      ["Range", `${fmt(s.range_h / 1000, 2)} km`],
      ["Weather", weatherNote(s)],
      ["Term", s.term || "—", s.success ? "good" : s.terminated ? "bad" : undefined],
      ["Box", landBoxNote(s), s.success ? "good" : s.terminated && !s.success ? "warn" : undefined],
      ["Breakup", s.destroy_reason || "—", s.destroy_reason ? "bad" : undefined],
      ["Best fit", fmt(t.best_ever, 0)],
      ["σ", fmt(t.sigma, 3)],
      ["Net", `${t.hidden ?? 8} hid · ${t.n_weights ?? "—"} w${t.growths ? ` · +${t.growths}` : ""}`],
    ];
    row(guid, guidRows);

    const pop = t.population || 128;
    const landPct = Math.round((t.land_rate ?? 0) * 100);
    const liveN = t.live_n ?? 0;
    const liveLands = t.live_lands ?? 0;
    const liveImpact = t.live_impact ?? 0;
    const liveMiss = t.live_miss ?? 0;
    const lastLands = t.last_successes ?? 0;
    const lastImpact = t.last_impact ?? 0;
    const lastMiss = t.last_miss ?? 0;
    const plane = s.plane_lock ? "2D" : "6DOF";
    const mix = (t.mix_left ?? 0) > 0
      ? t.mix_hard_pad
        ? " · mix windy pad"
        : " · mix next"
      : "";
    meta.textContent = t.running
      ? `RTLS · train ${stageN}/${stageCount} ${stageLabel} (${plane})${mix}\nthis gen ${liveN}/${pop} · ${liveLands} true · ${liveImpact} impact · ${liveMiss} miss\nlast gen ${lastLands}/${pop} true (${landPct}%) · ${lastImpact} impact`
      : `RTLS · train ${stageN}/${stageCount} ${stageLabel} (${plane})\ngen ${t.generation} · idle · ${t.episodes} episodes`;
    paintPhaseDots(dots, t);

    paintSpark(t);
    const liveFits = viz?.live ? nums(viz.live.fitnesses) : [];
    const liveTerms = viz?.live?.terms || [];
    paintHist(
      liveFits.length ? liveFits : t.fitnesses || [],
      liveFits.length ? liveTerms : t.terms || [],
    );
  };

  document.querySelector("#btn-train")!.addEventListener("click", () => engine.start_training());
  document.querySelector("#btn-pause")!.addEventListener("click", () => engine.pause_training());
  const params = new URLSearchParams(location.search);
  const episodeSeed = () => {
    const q = params.get("seed");
    if (q != null && q !== "" && Number.isFinite(Number(q))) {
      return Number(q) >>> 0;
    }
    return (Math.random() * 1e9) >>> 0;
  };

  document.querySelector("#btn-reset")!.addEventListener("click", () => {
    engine.reset(episodeSeed());
    scene?.resetTrail();
  });
  document.querySelector("#tog-destroy")!.addEventListener("change", (e) => {
    engine.set_destruction((e.target as HTMLInputElement).checked);
  });
  document.querySelector("#rng-wind")!.addEventListener("input", (e) => {
    const v = Number((e.target as HTMLInputElement).value);
    engine.set_wind_scale(v);
    document.querySelector("#wind-val")!.textContent = `${v.toFixed(1)}×`;
  });
  const setCamUi = (mode: CamMode) => {
    document.querySelectorAll(".cam").forEach((x) => x.classList.remove("on"));
    document.querySelector(`.cam[data-cam="${mode}"]`)?.classList.add("on");
    scene?.setCamMode(mode);
  };
  document.querySelector("#btn-reset-brain")!.addEventListener("click", () => {
    engine.reset_brain?.();
    clearBrain();
    scene?.resetTrail();
    lastStageN = 0;
    setCamUi("pad");
  });
  document.querySelector("#btn-reset-phase")!.addEventListener("click", () => {
    engine.reset_latest_phase?.();
    persistBrain(engine);
    scene?.resetTrail();
  });
  document.querySelector("#tog-autopilot")?.addEventListener("change", (e) => {
    const on = (e.target as HTMLInputElement).checked;
    engine.set_autopilot?.(on);
    scene?.resetTrail();
    setCamUi(on ? "orbit" : camForStage(lastStageN || 1));
  });
  document.querySelector("#tog-storm")!.addEventListener("change", (e) => {
    engine.set_storm((e.target as HTMLInputElement).checked);
  });
  document.querySelector("#tog-shear")!.addEventListener("change", (e) => {
    engine.set_shear((e.target as HTMLInputElement).checked);
  });
  document.querySelectorAll<HTMLButtonElement>(".cam").forEach((b) => {
    b.addEventListener("click", () => {
      setCamUi(b.dataset.cam as CamMode);
    });
  });
  document.querySelectorAll<HTMLButtonElement>(".warp").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll(".warp").forEach((x) => x.classList.remove("on"));
      b.classList.add("on");
      engine.set_time_warp(Number(b.dataset.warp));
    });
  });

  const loop = (now: number) => {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    if (engine.is_training()) {
      if (engine.train_for_ms(12)) persistBrain(engine);
    }
    engine.step_display(dt);
    snap = readSnap(engine);
    const train = readTrain(engine);
    const stageN = train.stage_n || 1;
    if (stageN !== lastStageN && lastStageN !== 0) {
      scene?.resetTrail();
      if (!engine.autopilot?.()) {
        setCamUi(camForStage(stageN));
      }
    }
    lastStageN = stageN;
    if (typeof train.viz_stamp === "number" && train.viz_stamp !== lastVizStamp) {
      lastVizStamp = train.viz_stamp;
      vizCache = readViz(engine);
      scene?.applySwarm(vizCache);
    }
    if (scene) {
      scene.apply(snap);
      scene.tickDebris(dt);
      scene.render();
      if (navMap) scene.paintMap(navMap, snap);
    }
    paint(snap, train, vizCache);
    requestAnimationFrame(loop);
  };
  const windQ = params.get("wind");
  if (windQ != null && Number.isFinite(Number(windQ))) {
    const v = Number(windQ);
    engine.set_wind_scale(v);
    const slider = document.querySelector<HTMLInputElement>("#rng-wind");
    if (slider) slider.value = String(v);
    const label = document.querySelector("#wind-val");
    if (label) label.textContent = `${v.toFixed(1)}×`;
  }

  if (params.get("seed") != null) {
    engine.reset(episodeSeed());
    scene?.resetTrail();
    snap = readSnap(engine);
  }

  paint(snap, readTrain(engine), null);
  setCamUi("pad");
  if (params.get("train") !== "0") {
    engine.start_training();
  }
  requestAnimationFrame(loop);
}

main().catch((err) => {
  console.error(err);
  const el = document.querySelector("#mission-status");
  if (el) el.textContent = err instanceof Error ? err.message.slice(0, 48) : "INIT FAILED";
});
