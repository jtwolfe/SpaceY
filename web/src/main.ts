import init, { Engine } from "./pkg/spacey.js";
import { SceneApp, type CamMode } from "./scene";
import type { Snapshot, TrainInfo } from "./types";

function fmt(n: number, d = 1): string {
  if (!Number.isFinite(n)) return "—";
  return n.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });
}

function row(dl: HTMLElement, rows: [string, string, string?][]) {
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
  const engine = new Engine();
  let scene: SceneApp | null = null;
  try {
    scene = new SceneApp(document.querySelector("#view")!);
  } catch (err) {
    console.error("WebGL scene failed; HUD will still run.", err);
  }

  const veh = document.querySelector("#veh-hud")!;
  const aero = document.querySelector("#aero-hud")!;
  const guid = document.querySelector("#guid-hud")!;
  const bars = document.querySelector("#fit-bars")!;
  const meta = document.querySelector("#train-meta")!;
  const status = document.querySelector("#mission-status")!;

  let last = performance.now();
  let snap: Snapshot = JSON.parse(engine.snapshot_json());

  const paint = (s: Snapshot, t: TrainInfo) => {
    const term = s.terminated ? s.term.toUpperCase() : s.phase;
    status.textContent = term;
    status.classList.toggle("good", s.success);
    status.classList.toggle("bad", s.terminated && !s.success);

    const inertialKm = s.speed_inertial / 1000;
    const groundKm = s.speed / 1000;
    const speedStr =
      s.scenario === "leo"
        ? `${fmt(inertialKm, 2)} km/s in`
        : `${fmt(s.speed, 0)} m/s`;
    row(veh, [
      ["T+ s", fmt(s.t, 1)],
      ["Altitude", `${fmt(s.alt / 1000, 2)} km`],
      ["Speed", speedStr],
      ["Ground", `${fmt(groundKm, 2)} km/s`],
      ["Mach", fmt(s.mach, 2)],
      ["Fuel", `${fmt(s.fuel / 1000, 1)} t`],
      ["Throttle", `${fmt(s.throttle * 100, 0)}% × ${s.n_engines}`],
      ["Tilt", `${fmt(s.tilt_deg, 1)}°`],
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
    const periKm = Number.isFinite(s.periapsis_alt) ? s.periapsis_alt / 1000 : Infinity;
    row(guid, [
      ["Phase", s.phase],
      ["Scenario", s.scenario === "leo" ? "LEO deorbit" : "RTLS"],
      ["Range", `${fmt((s.scenario === "leo" ? s.range_gc : s.range_h) / 1000, 2)} km`],
      ["Periapsis", Number.isFinite(periKm) ? `${fmt(periKm, 0)} km` : "escape"],
      ["Weather", [s.weather_storm ? "storm" : null, s.weather_shear ? "shear" : null].filter(Boolean).join("+") || "fair"],
      ["Term", s.term || "—", s.success ? "good" : s.terminated ? "bad" : undefined],
      ["Breakup", s.destroy_reason || "—", s.destroy_reason ? "bad" : undefined],
      ["Best fit", fmt(t.best_ever, 0)],
    ]);

    meta.textContent = t.running
      ? `gen ${t.generation} · eval ${t.evaluating}/${t.population} · ${t.episodes} eps · ${t.last_successes} land`
      : `gen ${t.generation} · idle · ${t.episodes} episodes`;

    const fits = t.fitnesses.length ? t.fitnesses : [0];
    const lo = Math.min(...fits);
    const hi = Math.max(...fits);
    const span = Math.max(1, hi - lo);
    bars.innerHTML = fits
      .map((f, i) => {
        const h = 6 + (72 * (f - lo)) / span;
        return `<i class="${i === 0 ? "best" : ""}" style="height:${h}px" title="${f.toFixed(0)}"></i>`;
      })
      .join("");
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
  const sub = document.querySelector("#mission-sub")!;
  const setScenarioUi = (id: number) => {
    document.querySelectorAll(".scen").forEach((x) => x.classList.remove("on"));
    document.querySelector(`.scen[data-scenario="${id}"]`)?.classList.add("on");
    sub.textContent =
      id === 1
        ? "FALCON 9-CLASS · STAGE 1 · LEO DEORBIT"
        : "FALCON 9-CLASS · STAGE 1 · RTLS TRAINER";
  };
  document.querySelectorAll<HTMLButtonElement>(".scen").forEach((b) => {
    b.addEventListener("click", () => {
      const id = Number(b.dataset.scenario);
      engine.set_scenario(id);
      setScenarioUi(id);
      engine.reset(episodeSeed());
      scene?.resetTrail();
    });
  });
  document.querySelector("#tog-storm")!.addEventListener("change", (e) => {
    engine.set_storm((e.target as HTMLInputElement).checked);
  });
  document.querySelector("#tog-shear")!.addEventListener("change", (e) => {
    engine.set_shear((e.target as HTMLInputElement).checked);
  });
  document.querySelectorAll<HTMLButtonElement>(".cam").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll(".cam").forEach((x) => x.classList.remove("on"));
      b.classList.add("on");
      if (scene) scene.camMode = b.dataset.cam as CamMode;
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
      engine.train_for_ms(14);
    }
    engine.step_display(dt);
    snap = JSON.parse(engine.snapshot_json()) as Snapshot;
    const train = JSON.parse(engine.train_json()) as TrainInfo;
    if (scene) {
      scene.apply(snap);
      scene.tickDebris(dt);
      scene.render();
    }
    paint(snap, train);
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

  if (params.get("scenario") === "leo") {
    engine.set_scenario(1);
    setScenarioUi(1);
    engine.reset(episodeSeed());
    scene?.resetTrail();
    snap = JSON.parse(engine.snapshot_json()) as Snapshot;
  } else if (params.get("seed") != null) {
    engine.reset(episodeSeed());
    scene?.resetTrail();
    snap = JSON.parse(engine.snapshot_json()) as Snapshot;
  }

  paint(snap, JSON.parse(engine.train_json()));
  requestAnimationFrame(loop);
}

main().catch((err) => {
  console.error(err);
  const el = document.querySelector("#mission-status");
  if (el) el.textContent = err instanceof Error ? err.message.slice(0, 48) : "INIT FAILED";
});
