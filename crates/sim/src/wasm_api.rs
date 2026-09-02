//! JavaScript surface: one `Engine` owns the display sim and the trainer.

use crate::cmaes::Trainer;
use crate::guidance::N_WEIGHTS;
use crate::scenario::Scenario;
use crate::sim::Sim;
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
}

#[wasm_bindgen]
impl Engine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Engine {
        let seed = 7;
        let destroy = true;
        let wind_scale = 1.0;
        let scenario = Scenario::Rtls;
        let weather = Weather::default();
        Engine {
            display: Sim::new_with(seed, destroy, wind_scale, scenario, weather),
            trainer: Trainer::new(seed.wrapping_add(99), destroy, wind_scale),
            time_warp: 1.0,
            destroy,
            wind_scale,
            scenario,
            weather,
            seed,
            watch_best: true,
        }
    }

    fn rebuild_display(&mut self, seed: u32) {
        let mut sim = Sim::new_with(seed, self.destroy, self.wind_scale, self.scenario, self.weather);
        if self.watch_best {
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

    pub fn set_scenario(&mut self, id: u32) {
        let next = Scenario::from_id(id);
        if next == self.scenario {
            return;
        }
        self.scenario = next;
        self.trainer.scenario = next;
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

    /// Spend up to `budget_ms` evaluating CMA-ES candidates. After a generation
    /// completes, the display vehicle is reset onto the current champion.
    pub fn train_for_ms(&mut self, budget_ms: f64) -> bool {
        let finished = self.trainer.tick(budget_ms.max(1.0));
        if finished && self.watch_best {
            self.reset(self.seed.wrapping_add(self.trainer.generation));
        }
        finished
    }

    /// Step with native `adaptive_dt` until term or `max_s` of sim time.
    /// Used so a 250× browser coast matches `cargo` / unit tests.
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
            return;
        }
        // Orbital coast at 250× needs more than 4 s of sim per frame.
        // Always take a full adaptive_dt — trimming the last slice to
        // `remain` desynchronizes the 2800 s LEO coast from native
        // (seed 88 landed at 20 m native, missed by 550 m in the
        // browser at 250×).
        let cap = if self.scenario.is_orbital() { 12.0 } else { 4.0 };
        let budget = (dt * self.time_warp).clamp(0.0, cap);
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

    pub fn train_json(&self) -> String {
        serde_json::to_string(&self.trainer.info()).unwrap_or_else(|_| "{}".into())
    }

    pub fn n_weights(&self) -> u32 {
        N_WEIGHTS as u32
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}
