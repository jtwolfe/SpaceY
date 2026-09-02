//! JavaScript surface: one `Engine` owns the display sim and the trainer.

use crate::cmaes::Trainer;
use crate::guidance::N_WEIGHTS;
use crate::sim::Sim;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Engine {
    display: Sim,
    trainer: Trainer,
    time_warp: f64,
    destroy: bool,
    wind_scale: f64,
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
        Engine {
            display: Sim::new(seed, destroy, wind_scale),
            trainer: Trainer::new(seed.wrapping_add(99), destroy, wind_scale),
            time_warp: 1.0,
            destroy,
            wind_scale,
            seed,
            watch_best: true,
        }
    }

    pub fn reset(&mut self, seed: u32) {
        self.seed = seed;
        let mut sim = Sim::new(seed, self.destroy, self.wind_scale);
        if self.watch_best {
            sim.set_weights(self.trainer.best_weights());
        }
        self.display = sim;
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

    pub fn set_time_warp(&mut self, warp: f64) {
        self.time_warp = warp.clamp(0.25, 200.0);
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

    /// Advance the display episode by `dt` seconds of *scene* time (already
    /// includes the caller's frame Δt; warp is applied here).
    pub fn step_display(&mut self, dt: f64) {
        if self.display.terminated() {
            return;
        }
        let mut remain = (dt * self.time_warp).clamp(0.0, 4.0);
        while remain > 1e-4 && !self.display.terminated() {
            let h = self.display.adaptive_dt().min(remain);
            self.display.step(h);
            remain -= h;
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
