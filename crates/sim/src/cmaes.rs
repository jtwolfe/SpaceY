//! Separable / full-lite CMA-ES trainer for the residual policy.
//!
//! Classic Hansen CMA-ES with Cholesky sampling and a z-space evolution path
//! for step-size. Covariance is rank-μ + rank-1. Designed to run inside the
//! browser: one generation is a handful of 6DOF rollouts, no backprop.

use crate::guidance::{TermReason, N_WEIGHTS};
use crate::math::{cos, exp, ln, sqrt};
use crate::scenario::Scenario;
use crate::sim::run_episode_with;
use crate::wind::Weather;
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use serde::Serialize;

const LAMBDA: usize = 12;
const MU: usize = 6;

#[derive(Clone, Debug, Serialize)]
pub struct CandidateStat {
    pub fitness: f64,
    pub term: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrainInfo {
    pub running: bool,
    pub generation: u32,
    pub evaluating: usize,
    pub population: usize,
    pub best_fitness: f64,
    pub mean_fitness: f64,
    pub best_ever: f64,
    pub sigma: f64,
    pub episodes: u32,
    pub fitnesses: Vec<f64>,
    pub terms: Vec<String>,
    pub last_successes: u32,
}

pub struct Trainer {
    pub running: bool,
    pub generation: u32,
    pub episodes: u32,
    pub destroy: bool,
    pub wind_scale: f64,
    pub scenario: Scenario,
    pub weather: Weather,
    pub scenario_seed: u32,
    mean: Vec<f64>,
    sigma: f64,
    /// Cholesky factor L of C (lower), row-major n×n.
    l: Vec<f64>,
    c: Vec<f64>,
    pc: Vec<f64>,
    ps: Vec<f64>,
    weights_w: Vec<f64>,
    mu_eff: f64,
    best_ever: f64,
    best_weights: Vec<f64>,
    last_fitnesses: Vec<f64>,
    last_terms: Vec<String>,
    last_successes: u32,
    rng: SmallRng,
    /// In-progress generation samples (weights).
    pending: Vec<Vec<f64>>,
    pending_z: Vec<Vec<f64>>,
    pending_y: Vec<Vec<f64>>,
    pending_f: Vec<Option<(f64, TermReason)>>,
    eval_index: usize,
}

impl Trainer {
    pub fn new(seed: u32, destroy: bool, wind_scale: f64) -> Self {
        let n = N_WEIGHTS;
        let mut weights_w = Vec::with_capacity(MU);
        for i in 0..MU {
            weights_w.push((ln(MU as f64 + 0.5) - ln((i + 1) as f64)).max(0.01));
        }
        let sw: f64 = weights_w.iter().sum();
        for w in weights_w.iter_mut() {
            *w /= sw;
        }
        let mu_eff = 1.0 / weights_w.iter().map(|w| w * w).sum::<f64>();
        Self {
            running: false,
            generation: 0,
            episodes: 0,
            destroy,
            wind_scale,
            scenario: Scenario::Rtls,
            weather: Weather::default(),
            scenario_seed: seed,
            mean: vec![0.0; n],
            sigma: 0.22,
            l: identity(n),
            c: identity(n),
            pc: vec![0.0; n],
            ps: vec![0.0; n],
            weights_w,
            mu_eff,
            best_ever: f64::NEG_INFINITY,
            best_weights: vec![0.0; n],
            last_fitnesses: vec![0.0; LAMBDA],
            last_terms: vec![String::new(); LAMBDA],
            last_successes: 0,
            rng: SmallRng::seed_from_u64(seed as u64 + 12345),
            pending: Vec::new(),
            pending_z: Vec::new(),
            pending_y: Vec::new(),
            pending_f: Vec::new(),
            eval_index: 0,
        }
    }

    pub fn best_weights(&self) -> &[f64] {
        &self.best_weights
    }

    pub fn start(&mut self) {
        self.running = true;
        if self.pending.is_empty() {
            self.sample_generation();
        }
    }

    pub fn pause(&mut self) {
        self.running = false;
    }

    fn sample_generation(&mut self) {
        let n = N_WEIGHTS;
        self.pending.clear();
        self.pending_z.clear();
        self.pending_y.clear();
        self.pending_f = vec![None; LAMBDA];
        self.eval_index = 0;
        for _ in 0..LAMBDA {
            let z: Vec<f64> = (0..n).map(|_| std_norm(&mut self.rng)).collect();
            let y = chol_mul(&self.l, &z, n);
            let x: Vec<f64> = self
                .mean
                .iter()
                .zip(y.iter())
                .map(|(m, yi)| m + self.sigma * *yi)
                .collect();
            self.pending.push(x);
            self.pending_z.push(z);
            self.pending_y.push(y);
        }
    }

    /// Evaluate as many pending candidates as fit in `budget_ms` wall time.
    /// Returns true if a generation completed.
    pub fn tick(&mut self, budget_ms: f64) -> bool {
        if !self.running {
            return false;
        }
        if self.pending.is_empty() {
            self.sample_generation();
        }
        let t0 = now_ms();
        while self.eval_index < LAMBDA {
            if now_ms() - t0 > budget_ms {
                return false;
            }
            let seed = self
                .scenario_seed
                .wrapping_add(self.generation * 17)
                .wrapping_add(self.eval_index as u32 * 31)
                .wrapping_add(self.rng.gen::<u32>() % 8);
            let (fit, term, _) = run_episode_with(
                &self.pending[self.eval_index],
                seed,
                self.destroy,
                self.wind_scale,
                self.scenario,
                self.weather,
            );
            self.pending_f[self.eval_index] = Some((fit, term));
            self.eval_index += 1;
            self.episodes += 1;
        }
        self.finish_generation();
        true
    }

    fn finish_generation(&mut self) {
        let n = N_WEIGHTS;
        let mut order: Vec<usize> = (0..LAMBDA).collect();
        order.sort_by(|a, b| {
            let fa = self.pending_f[*a].map(|p| p.0).unwrap_or(f64::NEG_INFINITY);
            let fb = self.pending_f[*b].map(|p| p.0).unwrap_or(f64::NEG_INFINITY);
            fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal)
        });
        self.last_fitnesses = order
            .iter()
            .map(|i| self.pending_f[*i].map(|p| p.0).unwrap_or(0.0))
            .collect();
        self.last_terms = order
            .iter()
            .map(|i| {
                self.pending_f[*i]
                    .map(|p| p.1.as_str().to_string())
                    .unwrap_or_default()
            })
            .collect();
        self.last_successes = self
            .pending_f
            .iter()
            .filter(|p| matches!(p, Some((_, TermReason::Success))))
            .count() as u32;

        let best = self.last_fitnesses[0];
        if best > self.best_ever {
            self.best_ever = best;
            self.best_weights = self.pending[order[0]].clone();
        }

        let mut yw = vec![0.0; n];
        let mut zw = vec![0.0; n];
        let mut mw = vec![0.0; n];
        for (k, &i) in order.iter().take(MU).enumerate() {
            let w = self.weights_w[k];
            for j in 0..n {
                yw[j] += w * self.pending_y[i][j];
                zw[j] += w * self.pending_z[i][j];
                mw[j] += w * self.pending[i][j];
            }
        }
        self.mean = mw;

        let c_sigma = (self.mu_eff + 2.0) / (n as f64 + self.mu_eff + 5.0);
        let d_sigma = 1.0
            + 2.0 * (0.0f64).max(sqrt((self.mu_eff - 1.0) / (n as f64 + 1.0)) - 1.0)
            + c_sigma;
        let c_c = (4.0 + self.mu_eff / n as f64) / (n as f64 + 4.0 + 2.0 * self.mu_eff / n as f64);
        let c_1 = 2.0 / ((n as f64 + 1.3).powi(2) + self.mu_eff);
        let c_mu = (1.0 - c_1)
            .min(2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((n as f64 + 2.0).powi(2) + self.mu_eff));

        for j in 0..n {
            self.ps[j] = (1.0 - c_sigma) * self.ps[j]
                + (sqrt(c_sigma * (2.0 - c_sigma) * self.mu_eff)) * zw[j];
        }
        let ps_norm = l2(&self.ps);
        let chi_n = sqrt(n as f64) * (1.0 - 1.0 / (4.0 * n as f64) + 1.0 / (21.0 * (n as f64).powi(2)));
        self.sigma *= exp((c_sigma / d_sigma) * (ps_norm / chi_n - 1.0));
        self.sigma = self.sigma.clamp(0.02, 1.4);

        for j in 0..n {
            self.pc[j] = (1.0 - c_c) * self.pc[j]
                + (sqrt(c_c * (2.0 - c_c) * self.mu_eff)) * yw[j];
        }

        // C ← (1 − c1 − cμ) C + c1 pc pcᵀ + cμ Σ w y yᵀ
        let decay = 1.0 - c_1 - c_mu;
        for i in 0..n {
            for j in 0..=i {
                let idx = i * n + j;
                let mut v = decay * self.c[idx] + c_1 * self.pc[i] * self.pc[j];
                for (k, &oi) in order.iter().take(MU).enumerate() {
                    v += c_mu * self.weights_w[k] * self.pending_y[oi][i] * self.pending_y[oi][j];
                }
                self.c[idx] = v;
                self.c[j * n + i] = v;
            }
            self.c[i * n + i] += 1e-10;
        }
        if let Some(l) = cholesky(&self.c, n) {
            self.l = l;
        }

        self.generation += 1;
        self.sample_generation();
    }

    pub fn info(&self) -> TrainInfo {
        let mean = if self.last_fitnesses.is_empty() {
            0.0
        } else {
            self.last_fitnesses.iter().sum::<f64>() / self.last_fitnesses.len() as f64
        };
        TrainInfo {
            running: self.running,
            generation: self.generation,
            evaluating: self.eval_index,
            population: LAMBDA,
            best_fitness: self.last_fitnesses.first().copied().unwrap_or(0.0),
            mean_fitness: mean,
            best_ever: if self.best_ever.is_finite() {
                self.best_ever
            } else {
                0.0
            },
            sigma: self.sigma,
            episodes: self.episodes,
            fitnesses: self.last_fitnesses.clone(),
            terms: self.last_terms.clone(),
            last_successes: self.last_successes,
        }
    }
}

fn identity(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

fn chol_mul(l: &[f64], z: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..=i {
            s += l[i * n + j] * z[j];
        }
        y[i] = s;
    }
    y
}

fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 1e-18 {
                    return None;
                }
                l[i * n + j] = sqrt(s);
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    Some(l)
}

fn l2(v: &[f64]) -> f64 {
    sqrt(v.iter().map(|x| x * x).sum::<f64>())
}

fn std_norm(rng: &mut SmallRng) -> f64 {
    let u1 = rng.gen::<f64>().clamp(1e-12, 1.0);
    let u2 = rng.gen::<f64>();
    sqrt(-2.0 * ln(u1)) * cos(2.0 * std::f64::consts::PI * u2)
}

fn now_ms() -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        js_sys_now()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64() * 1000.0)
            .unwrap_or(0.0)
    }
}

#[cfg(target_arch = "wasm32")]
fn js_sys_now() -> f64 {
    // Avoid a js-sys dep: Date.now via wasm-bindgen.
    now_via_js()
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = Date, js_name = now)]
    fn now_via_js() -> f64;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario::Scenario;

    #[test]
    fn leo_generation_advances() {
        let mut t = Trainer::new(11, true, 0.0);
        t.scenario = Scenario::LeoDeorbit;
        t.start();
        let mut ticks = 0;
        while t.generation < 1 && ticks < 32 {
            t.tick(120_000.0);
            ticks += 1;
        }
        assert!(
            t.generation >= 1,
            "LEO CMA-ES should finish a generation, gen={} eps={}",
            t.generation,
            t.episodes
        );
        assert!(t.episodes >= LAMBDA as u32);
        assert!(t.info().best_ever.is_finite());
    }
}
