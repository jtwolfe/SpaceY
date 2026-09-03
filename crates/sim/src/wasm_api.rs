//! JavaScript surface: one `Engine` owns the display sim and the trainer.

use crate::cmaes::Trainer;
use crate::policy::n_weights;
use crate::scenario::Scenario;
use crate::sim::{Pilot, Sim};
use crate::wind::Weather;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Engine {
    display: Sim,
    trainer: Trainer,
    time_warp: f64,
    destroy: bool,
    wind_scale: f64,
    scenario: Scenario,
    weather: Weather,
    seed: u32,
    watch_best: bool,
    autopilot: bool,
}

#[wasm_bindgen]
impl Engine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Engine {
        let seed = 7;
        let destroy = true;
        let wind_scale = 0.0;
        let scenario = Scenario::Pad;
        let weather = Weather::default();
        let mut display = Sim::new_with(seed, destroy, wind_scale, scenario, weather);
        display.pilot = Pilot::Policy;
        let mut trainer = Trainer::new(seed.wrapping_add(99), destroy, wind_scale);
        trainer.scenario = scenario;
        Engine {
            display,
            trainer,
            time_warp: 1.0,
            destroy,
            wind_scale,
            scenario,
            weather,
            seed,
            watch_best: true,
            autopilot: false,
        }
    }

    fn display_scenario(&self) -> Scenario {
        if self.autopilot {
            Scenario::Rtls
        } else {
            self.scenario
        }
    }

    fn rebuild_display(&mut self, seed: u32) {
        let mut sim = Sim::new_with(
            seed,
            self.destroy,
            self.wind_scale,
            self.display_scenario(),
            self.weather,
        );
        sim.pilot = if self.autopilot {
            Pilot::Autopilot
        } else {
            Pilot::Policy
        };
        if self.watch_best && !self.autopilot {
            sim.set_weights(self.trainer.best_weights());
        }
        self.display = sim;
    }

    pub fn reset(&mut self, seed: u32) {
        self.seed = seed;
        self.rebuild_display(seed);
    }

    pub fn set_destruction(&mut self, enabled: bool) {
        self.destroy = enabled;
        self.display.destroy_enabled = enabled;
        self.trainer.destroy = enabled;
    }

    pub fn set_wind_scale(&mut self, scale: f64) {
        let s = scale.clamp(0.0, 3.0);
        self.wind_scale = s;
        self.display.wind_scale = s;
        self.display.wind.scale = s;
        self.trainer.wind_scale = s;
    }

    /// Internal: keep champion weights when the trainer advances a stage.
    pub fn set_scenario(&mut self, id: u32) {
        let next = Scenario::from_id(id);
        if next == self.scenario {
            return;
        }
        self.scenario = next;
        self.trainer.scenario = next;
        self.trainer.weather = self.weather;
        self.trainer.destroy = self.destroy;
        self.trainer.wind_scale = self.wind_scale;
        self.trainer.retain_brain();
        self.rebuild_display(self.seed);
    }

    pub fn scenario(&self) -> u32 {
        self.scenario.id()
    }

    pub fn set_storm(&mut self, enabled: bool) {
        self.weather.storm = enabled;
        self.display.weather.storm = enabled;
        self.display.wind.weather.storm = enabled;
        self.trainer.weather.storm = enabled;
    }

    pub fn set_shear(&mut self, enabled: bool) {
        self.weather.shear = enabled;
        self.display.weather.shear = enabled;
        self.display.wind.weather.shear = enabled;
        self.trainer.weather.shear = enabled;
    }

    pub fn storm(&self) -> bool {
        self.weather.storm
    }

    pub fn shear(&self) -> bool {
        self.weather.shear
    }

    pub fn set_time_warp(&mut self, warp: f64) {
        self.time_warp = warp.clamp(0.25, 400.0);
    }

    pub fn time_warp(&self) -> f64 {
        self.time_warp
    }

    pub fn start_training(&mut self) {
        self.trainer.start();
    }

    pub fn pause_training(&mut self) {
        self.trainer.pause();
    }

    pub fn is_training(&self) -> bool {
        self.trainer.running
    }

    /// Spend up to `budget_ms` evaluating CMA-ES candidates.
    /// Display is *not* rewound each generation — the swarm view owns that.
    /// A new champion or a stage promote does restart the hero vehicle from T+0.
    pub fn train_for_ms(&mut self, budget_ms: f64) -> bool {
        let finished = self.trainer.tick(budget_ms.max(1.0));
        if self.trainer.take_promoted() {
            self.scenario = self.trainer.scenario;
            self.wind_scale = self.trainer.wind_scale;
            self.rebuild_display(self.seed.wrapping_add(self.trainer.generation));
        } else if self.trainer.took_new_best() && self.watch_best {
            self.reset(self.seed.wrapping_add(self.trainer.generation));
        }
        finished
    }

    /// Step with native `adaptive_dt` until term or `max_s` of sim time.
    pub fn fast_forward(&mut self, max_s: f64) {
        let t0 = self.display.t;
        let limit = max_s.clamp(0.0, 8_000.0);
        while !self.display.terminated() && self.display.t - t0 < limit {
            let h = self.display.adaptive_dt();
            self.display.step(h);
        }
    }

    pub fn seed(&self) -> u32 {
        self.seed
    }

    /// Advance the display episode by `dt` seconds of *scene* time (already
    /// includes the caller's frame Δt; warp is applied here).
    pub fn step_display(&mut self, dt: f64) {
        if self.display.terminated() {
            if self.trainer.running {
                self.rebuild_display(self.seed.wrapping_add(self.trainer.episodes + 1));
            } else {
                return;
            }
        }
        let budget = (dt * self.time_warp).clamp(0.0, 4.0);
        let mut used = 0.0;
        while used < budget && !self.display.terminated() {
            let h = self.display.adaptive_dt();
            self.display.step(h);
            used += h;
        }
    }

    pub fn snapshot_json(&self) -> String {
        serde_json::to_string(&self.display.snapshot()).unwrap_or_else(|_| "{}".into())
    }

    /// JS object snapshot — avoids `JSON.parse` of a string every frame.
    pub fn snapshot(&self) -> JsValue {
        let mut snap = self.display.snapshot();
        if !snap.periapsis_alt.is_finite() {
            snap.periapsis_alt = 1.0e12;
        }
        serde_wasm_bindgen::to_value(&snap).unwrap_or(JsValue::NULL)
    }

    pub fn train_json(&self) -> String {
        serde_json::to_string(&self.trainer.info()).unwrap_or_else(|_| "{}".into())
    }

    pub fn train_info(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.trainer.info()).unwrap_or(JsValue::NULL)
    }

    pub fn generation_viz(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.trainer.viz()).unwrap_or(JsValue::NULL)
    }

    pub fn set_autopilot(&mut self, enabled: bool) {
        self.autopilot = enabled;
        self.rebuild_display(self.seed);
    }

    pub fn autopilot(&self) -> bool {
        self.autopilot
    }

    pub fn n_weights(&self) -> u32 {
        n_weights(self.trainer.hidden()) as u32
    }
}

/// Zero-weight pad policy on the wasm32 numeric path. CI asserts this
/// does *not* land — the gym must not be a scripted skip.
#[wasm_bindgen]
pub fn run_hover_policy(seed: u32, destroy: bool, wind_scale: f64, max_steps: u32) -> String {
    use crate::sim::run_hover_policy_snapshot;
    serde_json::to_string(&run_hover_policy_snapshot(seed, destroy, wind_scale, max_steps))
        .unwrap_or_else(|_| "{}".into())
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}
