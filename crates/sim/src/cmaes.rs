//! Hansen **sep-CMA-ES** trainer for a frozen 8-hidden MLP (CMA-NeuroES).
//!
//! Diagonal covariance only. Population is large so the browser can draw a
//! whole generation at once (swarm trails). Topology does not grow.

use crate::constants::{MIX_GENS, RAJS_OWN_FRAC, SIGMA_LIVE_MIN};
use crate::guidance::TermReason;
use crate::math::{cos, exp, ln, sqrt};
use crate::policy::{
    apply_plane_lock_mask, n_weights, plane_lock_weight_mask, HIDDEN_START,
};
use crate::scenario::{Scenario, STAGE_COUNT};
use crate::sim::{run_episode_opts, EpisodeOpts, EpisodeTrace};
use crate::wind::{sample_weather_var, Weather};
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

pub const LAMBDA: usize = 128;
pub const MU: usize = 32;
const HISTORY_CAP: usize = 80;
const BRAIN_VERSION: u32 = 2;
const RESTART_GENS: u32 = 40;

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
    pub viz_stamp: u32,
    pub history_best: Vec<f32>,
    pub history_mean: Vec<f32>,
    pub history_lands: Vec<f32>,
    pub hidden: u32,
    pub n_weights: u32,
    pub growths: u32,
    pub restarts: u32,
    pub land_rate: f64,
    pub promote_ready: bool,
    pub promote_to: i32,
    pub mix_left: u32,
    pub mix_hard_pad: bool,
    pub stage: String,
    pub stage_n: u32,
    pub stage_count: u32,
    pub stage_label: String,
    pub live_n: u32,
    pub live_lands: u32,
    pub live_impact: u32,
    pub live_miss: u32,
    pub last_impact: u32,
    pub last_miss: u32,
    pub h_max: f64,
    pub h_frac: f64,
    pub slam_divert: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct PackedTrails {
    pub generation: u32,
    pub n: u32,
    pub best_idx: i32,
    pub fitnesses: Vec<f64>,
    pub terms: Vec<String>,
    pub t_end: Vec<f32>,
    pub xyz: Vec<f32>,
    pub lat: Vec<f32>,
    pub lon: Vec<f32>,
    pub counts: Vec<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct GenerationViz {
    pub stamp: u32,
    pub live: PackedTrails,
    pub prev: PackedTrails,
}

fn pack_trails(generation: u32, paths: &[EpisodeTrace]) -> PackedTrails {
    let mut best_idx: i32 = -1;
    let mut best = f64::NEG_INFINITY;
    let mut fitnesses = Vec::with_capacity(paths.len());
    let mut terms = Vec::with_capacity(paths.len());
    let mut t_end = Vec::with_capacity(paths.len());
    let mut xyz = Vec::new();
    let mut lat = Vec::new();
    let mut lon = Vec::new();
    let mut counts = Vec::with_capacity(paths.len());
    for (i, p) in paths.iter().enumerate() {
        fitnesses.push(p.fitness);
        terms.push(p.term.as_str().to_string());
        t_end.push(p.t_end);
        counts.push((p.lat.len() as u32).min(p.xyz.len() as u32 / 3));
        xyz.extend_from_slice(&p.xyz);
        lat.extend_from_slice(&p.lat);
        lon.extend_from_slice(&p.lon);
        if p.fitness > best {
            best = p.fitness;
            best_idx = i as i32;
        }
    }
    PackedTrails {
        generation,
        n: paths.len() as u32,
        best_idx,
        fitnesses,
        terms,
        t_end,
        xyz,
        lat,
        lon,
        counts,
    }
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
    /// sep-CMA: sqrt of the diagonal of C. Sampling is x = m + σ (d ⊙ z).
    d: Vec<f64>,
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
    pending: Vec<Vec<f64>>,
    pending_z: Vec<Vec<f64>>,
    pending_y: Vec<Vec<f64>>,
    pending_f: Vec<Option<(f64, TermReason)>>,
    eval_index: usize,
    live_paths: Vec<EpisodeTrace>,
    prev_paths: Vec<EpisodeTrace>,
    viz_stamp: u32,
    history_best: Vec<f32>,
    history_mean: Vec<f32>,
    history_lands: Vec<f32>,
    improved_best: bool,
    hidden: usize,
    gens_since_best: u32,
    restarts: u32,
    land_streak: u32,
    promote_ready: bool,
    promoted: bool,
    pub pin_storm: bool,
    pub pin_shear: bool,
    mix_left: u32,
    mix_next: Option<Scenario>,
    mix_hard_pad: bool,
    /// Champion that earned the last promote — used to undo the current stage.
    stage_anchor: Option<StageAnchor>,
    /// RAJS: 1 = guide may fly the whole timeout; 0 = net owns T+0.
    h_frac: f64,
    land_ema: f64,
    slam_divert: bool,
}

impl Trainer {
    pub fn new(seed: u32, destroy: bool, wind_scale: f64) -> Self {
        let n = n_weights(HIDDEN_START);
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
            scenario: Scenario::Pad,
            weather: Weather::default(),
            scenario_seed: seed,
            mean: vec![0.0; n],
            sigma: 0.28,
            d: vec![1.0; n],
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
            live_paths: Vec::new(),
            prev_paths: Vec::new(),
            viz_stamp: 0,
            history_best: Vec::new(),
            history_mean: Vec::new(),
            history_lands: Vec::new(),
            improved_best: false,
            hidden: HIDDEN_START,
            gens_since_best: 0,
            restarts: 0,
            land_streak: 0,
            promote_ready: false,
            promoted: false,
            pin_storm: false,
            pin_shear: false,
            mix_left: 0,
            mix_next: None,
            mix_hard_pad: false,
            stage_anchor: None,
            h_frac: 1.0,
            land_ema: 0.0,
            slam_divert: false,
        }
    }

    pub fn best_weights(&self) -> &[f64] {
        &self.best_weights
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    pub fn mix_hard_pad(&self) -> bool {
        self.mix_hard_pad && self.mix_left > 0
    }

    pub fn slam_divert(&self) -> bool {
        self.slam_divert
    }

    pub fn any_success(&self) -> bool {
        self.last_successes > 0 || self.live_paths.iter().any(|p| p.success)
    }

    fn dim(&self) -> usize {
        self.mean.len()
    }

    /// Drop CMA-ES state when the mission is reset.
    pub fn reset_policy(&mut self) {
        self.hidden = HIDDEN_START;
        self.restarts = 0;
        self.gens_since_best = 0;
        self.land_streak = 0;
        self.promote_ready = false;
        self.promoted = false;
        self.mix_left = 0;
        self.mix_next = None;
        self.mix_hard_pad = false;
        self.stage_anchor = None;
        self.h_frac = 1.0;
        self.land_ema = 0.0;
        self.slam_divert = false;
        let n = n_weights(self.hidden);
        self.running = false;
        self.generation = 0;
        self.episodes = 0;
        self.mean = vec![0.0; n];
        self.sigma = 0.28;
        self.d = vec![1.0; n];
        self.pc = vec![0.0; n];
        self.ps = vec![0.0; n];
        self.best_ever = f64::NEG_INFINITY;
        self.best_weights = vec![0.0; n];
        self.last_fitnesses = vec![0.0; LAMBDA];
        self.last_terms = vec![String::new(); LAMBDA];
        self.last_successes = 0;
        self.pending.clear();
        self.pending_z.clear();
        self.pending_y.clear();
        self.pending_f.clear();
        self.eval_index = 0;
        self.live_paths.clear();
        self.prev_paths.clear();
        self.viz_stamp = self.viz_stamp.wrapping_add(1);
        self.history_best.clear();
        self.history_mean.clear();
        self.history_lands.clear();
        self.improved_best = false;
    }

    /// Keep the MLP mean when moving along the curriculum; reset CMA covariance.
    pub fn retain_brain(&mut self) {
        let running = self.running;
        let n = self.dim();
        self.running = running;
        self.generation = 0;
        self.episodes = 0;
        self.sigma = (self.sigma * 1.15).max(SIGMA_LIVE_MIN).clamp(SIGMA_LIVE_MIN, 0.45);
        self.d = vec![1.0; n];
        self.pc = vec![0.0; n];
        self.ps = vec![0.0; n];
        self.best_ever = f64::NEG_INFINITY;
        self.last_fitnesses = vec![0.0; LAMBDA];
        self.last_terms = vec![String::new(); LAMBDA];
        self.last_successes = 0;
        self.pending.clear();
        self.pending_z.clear();
        self.pending_y.clear();
        self.pending_f.clear();
        self.eval_index = 0;
        self.live_paths.clear();
        self.prev_paths.clear();
        self.viz_stamp = self.viz_stamp.wrapping_add(1);
        self.history_best.clear();
        self.history_mean.clear();
        self.history_lands.clear();
        self.improved_best = false;
        self.gens_since_best = 0;
        self.land_streak = 0;
        self.promote_ready = false;
        self.mix_left = 0;
        self.mix_next = None;
        self.mix_hard_pad = false;
        self.h_frac = 1.0;
        self.land_ema = 0.0;
        if running {
            self.sample_generation();
        }
    }

    fn capture_stage_anchor(&mut self) {
        self.stage_anchor = Some(StageAnchor {
            weights: self.best_weights.clone(),
            mean: self.mean.clone(),
            hidden: self.hidden as u32,
            sigma: self.sigma,
        });
    }

    fn apply_stage_anchor(&mut self, anchor: &StageAnchor) -> bool {
        let hidden = HIDDEN_START;
        let n = n_weights(hidden);
        if anchor.weights.len() != n || anchor.mean.len() != n {
            return false;
        }
        self.hidden = hidden;
        self.best_weights = anchor.weights.clone();
        self.mean = anchor.mean.clone();
        self.sigma = anchor.sigma.max(SIGMA_LIVE_MIN).clamp(SIGMA_LIVE_MIN, 1.4);
        true
    }

    /// Wipe the net and CMA state, return to pad slam.
    pub fn reset_brain(&mut self) {
        let running = self.running;
        let destroy = self.destroy;
        let wind_scale = self.wind_scale;
        let pin_storm = self.pin_storm;
        let pin_shear = self.pin_shear;
        self.reset_policy();
        self.scenario = Scenario::Pad;
        self.destroy = destroy;
        self.wind_scale = wind_scale;
        self.pin_storm = pin_storm;
        self.pin_shear = pin_shear;
        if running {
            self.start();
        }
    }

    /// Undo training on the current stage. Restores the last promote snapshot
    /// when one exists; otherwise drops back one curriculum gate.
    pub fn reset_latest_phase(&mut self) {
        let running = self.running;
        if let Some(anchor) = self.stage_anchor.clone() {
            if self.apply_stage_anchor(&anchor) {
                let sigma = self.sigma;
                self.retain_brain();
                self.sigma = sigma;
                if running && self.pending.is_empty() {
                    self.sample_generation();
                }
                return;
            }
        }
        if let Some(prev) = self.scenario.prev_gate() {
            self.scenario = prev;
            self.stage_anchor = None;
            self.retain_brain();
            return;
        }
        self.reset_brain();
    }

    /// Advance one training stage, keep champion weights, reset covariance.
    fn commit_promote(&mut self) -> bool {
        let Some(next) = self.mix_next.take() else {
            self.promote_ready = false;
            return false;
        };
        self.capture_stage_anchor();
        self.mix_left = 0;
        self.mix_hard_pad = false;
        self.scenario = next;
        if next == Scenario::Slam {
            self.slam_divert = false;
        }
        if next == Scenario::Wind && self.wind_scale < 0.4 {
            self.wind_scale = 1.0;
        }
        self.promoted = true;
        self.retain_brain();
        if next == Scenario::Attitude {
            self.inflate_unlocked_axes();
        }
        true
    }

    fn inflate_unlocked_axes(&mut self) {
        let mask = plane_lock_weight_mask(self.hidden);
        for (i, freeze) in mask.iter().enumerate() {
            if i >= self.d.len() {
                break;
            }
            if *freeze {
                // Newly unmasked 6DOF slices: start at 0, not tanh rails, with
                // a slightly larger diagonal so CMA actually searches them.
                self.mean[i] = 0.0;
                self.best_weights[i] = 0.0;
                self.d[i] = 1.25;
            }
        }
    }

    pub fn take_promoted(&mut self) -> bool {
        let p = self.promoted;
        self.promoted = false;
        p
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
        let n = self.dim();
        self.pending.clear();
        self.pending_z.clear();
        self.pending_y.clear();
        self.pending_f = vec![None; LAMBDA];
        self.eval_index = 0;
        let lock = self.scenario.plane_lock();
        for _ in 0..LAMBDA {
            let mut z: Vec<f64> = (0..n).map(|_| std_norm(&mut self.rng)).collect();
            let mut y = vec![0.0; n];
            let mut x = vec![0.0; n];
            for i in 0..n {
                y[i] = self.d[i] * z[i];
                x[i] = self.mean[i] + self.sigma * y[i];
            }
            if lock {
                apply_plane_lock_mask(&mut x, self.hidden);
                let mask = plane_lock_weight_mask(self.hidden);
                for (i, freeze) in mask.iter().enumerate() {
                    if *freeze {
                        y[i] = 0.0;
                        z[i] = 0.0;
                    }
                }
            }
            self.pending.push(x);
            self.pending_z.push(z);
            self.pending_y.push(y);
        }
        if lock {
            apply_plane_lock_mask(&mut self.mean, self.hidden);
        }
    }

    /// Evaluate as many pending candidates as fit in `budget_ms` wall time.
    /// Returns true if a generation completed.
    pub fn tick(&mut self, budget_ms: f64) -> bool {
        self.improved_best = false;
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
            let pad_harden = self.mix_hard_pad && self.mix_left > 0;
            let hard_pad = pad_harden && self.rng.gen::<bool>();
            let ep_scen = if pad_harden {
                self.scenario
            } else if self.mix_left > 0 {
                if self.rng.gen::<bool>() {
                    self.scenario
                } else {
                    self.mix_next.unwrap_or(self.scenario)
                }
            } else {
                self.scenario
            };
            let timeout = ep_scen.timeout();
            let guide_until = self.rng.gen::<f64>() * self.h_frac * timeout;
            let (wx, wind) = sample_weather_var(
                ep_scen,
                self.wind_scale.max(0.0),
                self.pin_storm,
                self.pin_shear,
                hard_pad,
                &mut self.rng,
            );
            self.weather = wx;
            let tr = run_episode_opts(
                &self.pending[self.eval_index],
                seed,
                self.destroy,
                wind,
                ep_scen,
                wx,
                true,
                EpisodeOpts {
                    hard_pad,
                    slam_divert: self.slam_divert || ep_scen != Scenario::Slam,
                    guide_until,
                },
            );
            self.pending_f[self.eval_index] = Some((tr.fitness, tr.term));
            self.live_paths.push(tr);
            self.eval_index += 1;
            self.episodes += 1;
            self.viz_stamp = self.viz_stamp.wrapping_add(1);
        }
        self.finish_generation();
        true
    }

    pub fn viz(&self) -> GenerationViz {
        GenerationViz {
            stamp: self.viz_stamp,
            live: pack_trails(self.generation, &self.live_paths),
            prev: pack_trails(
                self.generation.saturating_sub(1),
                &self.prev_paths,
            ),
        }
    }

    /// True if the last `tick` that finished a generation found a new champion.
    pub fn took_new_best(&self) -> bool {
        self.improved_best
    }

    fn finish_generation(&mut self) {
        let n = self.dim();
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
        let mut pos_lands = 0u32;
        let mut neg_lands = 0u32;
        for p in &self.live_paths {
            if p.success {
                if p.spawn_ve >= 0.0 {
                    pos_lands += 1;
                } else {
                    neg_lands += 1;
                }
            }
        }
        let land_frac = self.last_successes as f64 / LAMBDA as f64;
        self.land_ema = 0.75 * self.land_ema + 0.25 * land_frac;
        if self.land_ema >= 0.22 {
            self.h_frac = (self.h_frac * 0.90).max(0.0);
        } else if land_frac < 0.05 {
            self.h_frac = (self.h_frac + 0.06).min(1.0);
        }
        let net_owns = self.h_frac <= RAJS_OWN_FRAC + 1e-9;
        let both_signs = !(self.scenario == Scenario::Slam && self.slam_divert)
            || (pos_lands >= 1 && neg_lands >= 1);
        let was_mixing = self.mix_left > 0;
        if !was_mixing {
            if land_frac + 1e-9 >= 0.30 && net_owns {
                self.land_streak = self.land_streak.saturating_add(1);
            } else {
                self.land_streak = 0;
                self.promote_ready = false;
            }
            if self.land_streak >= 3 && net_owns {
                if self.scenario == Scenario::Slam && !self.slam_divert {
                    self.slam_divert = true;
                    self.land_streak = 0;
                    self.h_frac = self.h_frac.max(0.55);
                    self.promote_ready = false;
                } else if both_signs {
                    if let Some(next) = self.scenario.next_gate() {
                        self.promote_ready = true;
                        self.mix_next = Some(next);
                        self.mix_left = MIX_GENS;
                        self.mix_hard_pad = self.scenario == Scenario::Pad;
                    }
                }
            }
        }
        let opened_mix = !was_mixing && self.mix_left > 0;

        let best = self.last_fitnesses[0];
        if best > self.best_ever {
            self.best_ever = best;
            self.best_weights = self.pending[order[0]].clone();
            self.improved_best = true;
            self.gens_since_best = 0;
        } else {
            self.gens_since_best = self.gens_since_best.saturating_add(1);
        }

        let mean = self.last_fitnesses.iter().sum::<f64>() / self.last_fitnesses.len().max(1) as f64;
        self.history_best.push(self.best_ever as f32);
        self.history_mean.push(mean as f32);
        self.history_lands.push(self.last_successes as f32);
        if self.history_best.len() > HISTORY_CAP {
            self.history_best.remove(0);
            self.history_mean.remove(0);
            self.history_lands.remove(0);
        }

        self.prev_paths = std::mem::take(&mut self.live_paths);
        self.viz_stamp = self.viz_stamp.wrapping_add(1);

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
        if self.scenario.plane_lock() {
            apply_plane_lock_mask(&mut self.mean, self.hidden);
        }

        let n_f = n as f64;
        let c_sigma = (self.mu_eff + 2.0) / (n_f + self.mu_eff + 5.0);
        let d_sigma = 1.0
            + 2.0 * (0.0f64).max(sqrt((self.mu_eff - 1.0) / (n_f + 1.0)) - 1.0)
            + c_sigma;
        let c_c = (4.0 + self.mu_eff / n_f) / (n_f + 4.0 + 2.0 * self.mu_eff / n_f);
        let mut c_1 = 2.0 / ((n_f + 1.3).powi(2) + self.mu_eff);
        let mut c_mu = (1.0 - c_1).min(
            2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff)
                / ((n_f + 2.0).powi(2) + self.mu_eff),
        );
        // Ros & Hansen 2008: diagonal C learns ~n times fewer entries.
        let sep = (n_f + 2.0) / 3.0;
        c_1 = (c_1 * sep).min(1.0);
        c_mu = (c_mu * sep).min(1.0 - c_1);

        let mask = if self.scenario.plane_lock() {
            plane_lock_weight_mask(self.hidden)
        } else {
            vec![false; n]
        };

        for j in 0..n {
            if mask[j] {
                self.ps[j] = 0.0;
                continue;
            }
            self.ps[j] = (1.0 - c_sigma) * self.ps[j]
                + (sqrt(c_sigma * (2.0 - c_sigma) * self.mu_eff)) * zw[j];
        }
        let ps_norm = l2(&self.ps);
        let chi_n = sqrt(n_f) * (1.0 - 1.0 / (4.0 * n_f) + 1.0 / (21.0 * n_f.powi(2)));
        self.sigma *= exp((c_sigma / d_sigma) * (ps_norm / chi_n - 1.0));
        self.sigma = self.sigma.clamp(0.02, 1.4);

        for j in 0..n {
            if mask[j] {
                self.pc[j] = 0.0;
                self.d[j] = 1.0;
                continue;
            }
            self.pc[j] = (1.0 - c_c) * self.pc[j]
                + (sqrt(c_c * (2.0 - c_c) * self.mu_eff)) * yw[j];
            let mut y2 = 0.0;
            for (k, &oi) in order.iter().take(MU).enumerate() {
                let yi = self.pending_y[oi][j];
                y2 += self.weights_w[k] * yi * yi;
            }
            let cii = (1.0 - c_1 - c_mu) * self.d[j] * self.d[j]
                + c_1 * self.pc[j] * self.pc[j]
                + c_mu * y2;
            self.d[j] = sqrt(cii.max(1e-12)).clamp(1e-4, 10.0);
        }

        self.generation += 1;
        self.maybe_restart();
        if opened_mix {
            self.sample_generation();
        } else if was_mixing {
            self.mix_left = self.mix_left.saturating_sub(1);
            if self.mix_left == 0 {
                self.commit_promote();
            } else {
                self.sample_generation();
            }
        } else {
            self.sample_generation();
        }
    }

    fn maybe_restart(&mut self) {
        if self.gens_since_best < RESTART_GENS {
            return;
        }
        if self.sigma > 0.05 {
            return;
        }
        let n = self.dim();
        self.sigma = 0.28;
        self.d = vec![1.0; n];
        self.pc = vec![0.0; n];
        self.ps = vec![0.0; n];
        self.gens_since_best = 0;
        self.restarts += 1;
        if self.scenario.plane_lock() {
            apply_plane_lock_mask(&mut self.mean, self.hidden);
        }
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
            best_ever: {
                let live_best = self
                    .live_paths
                    .iter()
                    .map(|p| p.fitness)
                    .fold(f64::NEG_INFINITY, f64::max);
                let b = if self.best_ever.is_finite() {
                    self.best_ever.max(live_best)
                } else {
                    live_best
                };
                if b.is_finite() { b } else { 0.0 }
            },
            sigma: self.sigma,
            episodes: self.episodes,
            fitnesses: self.last_fitnesses.clone(),
            terms: self.last_terms.clone(),
            last_successes: self.last_successes,
            viz_stamp: self.viz_stamp,
            history_best: self.history_best.clone(),
            history_mean: self.history_mean.clone(),
            history_lands: self.history_lands.clone(),
            hidden: self.hidden as u32,
            n_weights: self.dim() as u32,
            growths: 0,
            restarts: self.restarts,
            land_rate: if self.last_fitnesses.is_empty() {
                0.0
            } else {
                self.last_successes as f64 / LAMBDA as f64
            },
            promote_ready: self.promote_ready,
            promote_to: self
                .scenario
                .next_gate()
                .map(|s| s.id() as i32)
                .unwrap_or(-1),
            mix_left: self.mix_left,
            mix_hard_pad: self.mix_hard_pad,
            stage: self.scenario.as_str().to_string(),
            stage_n: self.scenario.id() + 1,
            stage_count: STAGE_COUNT,
            stage_label: self.scenario.label().to_string(),
            live_n: self.live_paths.len() as u32,
            live_lands: count_term_paths(&self.live_paths, TermReason::Success),
            live_impact: count_term_paths(&self.live_paths, TermReason::Destroyed),
            live_miss: count_term_paths(&self.live_paths, TermReason::GroundMiss),
            last_impact: count_term_str(&self.last_terms, TermReason::Destroyed.as_str()),
            last_miss: count_term_str(&self.last_terms, TermReason::GroundMiss.as_str()),
            h_max: self.h_frac * self.scenario.timeout(),
            h_frac: self.h_frac,
            slam_divert: self.slam_divert,
        }
    }

    pub fn export_brain(&self) -> String {
        let blob = BrainBlob {
            v: BRAIN_VERSION,
            weights: self.best_weights.clone(),
            mean: self.mean.clone(),
            hidden: self.hidden as u32,
            stage: self.scenario.id(),
            generation: self.generation,
            best_ever: if self.best_ever.is_finite() {
                self.best_ever
            } else {
                0.0
            },
            sigma: self.sigma,
            anchor: self.stage_anchor.clone(),
            h_frac: self.h_frac,
            slam_divert: self.slam_divert,
        };
        serde_json::to_string(&blob).unwrap_or_else(|_| "{}".into())
    }

    pub fn import_brain(&mut self, json: &str) -> bool {
        let Ok(blob) = serde_json::from_str::<BrainBlob>(json) else {
            return false;
        };
        if blob.v != BRAIN_VERSION {
            return false;
        }
        if blob.hidden as usize != HIDDEN_START {
            return false;
        }
        let n = n_weights(HIDDEN_START);
        if blob.weights.len() != n || blob.mean.len() != n {
            return false;
        }
        self.hidden = HIDDEN_START;
        self.best_weights = blob.weights;
        self.mean = blob.mean;
        self.scenario = Scenario::from_id(blob.stage);
        self.generation = blob.generation;
        self.best_ever = blob.best_ever;
        self.sigma = blob.sigma.max(SIGMA_LIVE_MIN).clamp(SIGMA_LIVE_MIN, 1.4);
        self.d = vec![1.0; n];
        self.pc = vec![0.0; n];
        self.ps = vec![0.0; n];
        self.mix_left = 0;
        self.mix_next = None;
        self.mix_hard_pad = false;
        self.promote_ready = false;
        self.h_frac = blob.h_frac.clamp(0.0, 1.0);
        self.slam_divert = blob.slam_divert;
        self.pending.clear();
        self.pending_z.clear();
        self.pending_y.clear();
        self.pending_f.clear();
        self.eval_index = 0;
        self.stage_anchor = blob.anchor.filter(|a| {
            let an = n_weights(HIDDEN_START);
            a.weights.len() == an && a.mean.len() == an
        });
        true
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BrainBlob {
    v: u32,
    weights: Vec<f64>,
    mean: Vec<f64>,
    hidden: u32,
    stage: u32,
    generation: u32,
    best_ever: f64,
    sigma: f64,
    #[serde(default)]
    anchor: Option<StageAnchor>,
    #[serde(default = "default_h_frac")]
    h_frac: f64,
    #[serde(default)]
    slam_divert: bool,
}

fn default_h_frac() -> f64 {
    1.0
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StageAnchor {
    weights: Vec<f64>,
    mean: Vec<f64>,
    hidden: u32,
    sigma: f64,
}

fn count_term_paths(paths: &[EpisodeTrace], want: TermReason) -> u32 {
    paths.iter().filter(|p| p.term == want).count() as u32
}

fn count_term_str(terms: &[String], want: &str) -> u32 {
    terms.iter().filter(|t| t.as_str() == want).count() as u32
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
    fn pad_generation_advances() {
        let mut t = Trainer::new(11, true, 0.0);
        t.scenario = Scenario::Pad;
        t.start();
        let mut ticks = 0;
        while t.generation < 1 && ticks < 400 {
            t.tick(120_000.0);
            ticks += 1;
        }
        assert!(
            t.generation >= 1,
            "CMA-ES should finish a generation, gen={} eps={}",
            t.generation,
            t.episodes
        );
        assert!(t.episodes >= LAMBDA as u32);
        assert!(t.info().best_ever.is_finite());
        assert_eq!(t.viz().prev.n, LAMBDA as u32);
    }

    #[test]
    fn reset_policy_zeros_champion() {
        let mut t = Trainer::new(1, true, 0.0);
        t.best_weights[0] = 3.0;
        t.generation = 9;
        t.episodes = 120;
        t.reset_policy();
        assert_eq!(t.generation, 0);
        assert_eq!(t.episodes, 0);
        assert!(!t.running);
        assert!(t.best_weights.iter().all(|w| *w == 0.0));
    }

    #[test]
    fn reset_brain_returns_to_pad() {
        let mut t = Trainer::new(1, true, 1.0);
        t.scenario = Scenario::Attitude;
        t.best_weights[0] = 0.8;
        t.running = true;
        t.reset_brain();
        assert_eq!(t.scenario, Scenario::Pad);
        assert!(t.best_weights.iter().all(|w| *w == 0.0));
        assert!(t.running);
        assert!((t.wind_scale - 1.0).abs() < 1e-12);
    }

    #[test]
    fn reset_latest_phase_restores_promote_snapshot() {
        let mut t = Trainer::new(1, true, 0.0);
        t.best_weights[0] = 0.41;
        t.mean[0] = 0.41;
        t.mix_next = Some(Scenario::Attitude);
        assert!(t.commit_promote());
        assert_eq!(t.scenario, Scenario::Attitude);
        t.best_weights[0] = 0.99;
        t.mean[0] = 0.99;
        t.reset_latest_phase();
        assert_eq!(t.scenario, Scenario::Attitude);
        assert!((t.best_weights[0] - 0.41).abs() < 1e-12);
        assert!((t.mean[0] - 0.41).abs() < 1e-12);
    }

    #[test]
    fn reset_latest_phase_without_snapshot_demotes() {
        let mut t = Trainer::new(1, true, 0.0);
        t.scenario = Scenario::Attitude;
        t.best_weights[0] = 0.2;
        t.mean[0] = 0.2;
        t.reset_latest_phase();
        assert_eq!(t.scenario, Scenario::Slam);
        assert!((t.best_weights[0] - 0.2).abs() < 1e-12);
    }

    #[test]
    fn retain_brain_keeps_mean() {
        let mut t = Trainer::new(1, true, 0.0);
        t.mean[0] = 0.42;
        t.best_weights[0] = 0.42;
        t.generation = 5;
        t.retain_brain();
        assert_eq!(t.generation, 0);
        assert!((t.mean[0] - 0.42).abs() < 1e-12);
        assert!((t.best_weights[0] - 0.42).abs() < 1e-12);
        assert!(!t.promote_ready);
    }

    #[test]
    fn import_brain_roundtrip() {
        let mut t = Trainer::new(1, true, 1.0);
        t.best_weights[0] = 0.31;
        t.mean[0] = 0.31;
        t.generation = 4;
        t.capture_stage_anchor();
        let json = t.export_brain();
        let mut u = Trainer::new(2, true, 1.0);
        assert!(u.import_brain(&json));
        assert!((u.best_weights[0] - 0.31).abs() < 1e-12);
        assert_eq!(u.generation, 4);
        u.best_weights[0] = 0.9;
        u.reset_latest_phase();
        assert!((u.best_weights[0] - 0.31).abs() < 1e-12);
        assert!(!u.import_brain("{}"));
        let n = n_weights(HIDDEN_START);
        let v1 = serde_json::json!({
            "v": 1,
            "weights": vec![0.0; n],
            "mean": vec![0.0; n],
            "hidden": 8,
            "stage": 0,
            "generation": 0,
            "best_ever": 0.0,
            "sigma": 0.3,
        });
        assert!(!u.import_brain(&v1.to_string()));
        let grown = serde_json::json!({
            "v": 2,
            "weights": vec![0.0; n],
            "mean": vec![0.0; n],
            "hidden": 16,
            "stage": 0,
            "generation": 0,
            "best_ever": 0.0,
            "sigma": 0.3,
        });
        assert!(!u.import_brain(&grown.to_string()));
    }

    #[test]
    fn hidden_stays_frozen() {
        let t = Trainer::new(1, true, 0.0);
        assert_eq!(t.hidden, 8);
        assert_eq!(t.dim(), 210);
        assert_eq!(t.info().growths, 0);
        assert_eq!(HIDDEN_START, 8);
    }

    #[test]
    fn pad_samples_zero_locked_outputs() {
        let mut t = Trainer::new(1, true, 0.0);
        t.scenario = Scenario::Pad;
        t.start();
        let mask = crate::policy::plane_lock_weight_mask(8);
        for x in &t.pending {
            for (i, freeze) in mask.iter().enumerate() {
                if *freeze {
                    assert_eq!(x[i], 0.0);
                }
            }
        }
    }

    #[test]
    fn retain_and_import_keep_live_sigma() {
        let mut t = Trainer::new(1, true, 0.0);
        t.sigma = 0.05;
        t.retain_brain();
        assert!(t.sigma >= SIGMA_LIVE_MIN - 1e-12);
        t.sigma = 0.05;
        let json = t.export_brain();
        let mut u = Trainer::new(2, true, 0.0);
        assert!(u.import_brain(&json));
        assert!(u.sigma >= SIGMA_LIVE_MIN - 1e-12);
        assert!((u.h_frac - 1.0).abs() < 1e-12);
    }

    #[test]
    fn stuck_restart_does_not_grow_hidden() {
        let mut t = Trainer::new(1, true, 0.0);
        t.gens_since_best = RESTART_GENS;
        t.sigma = 0.04;
        t.maybe_restart();
        assert_eq!(t.hidden, 8);
        assert_eq!(t.dim(), 210);
        assert_eq!(t.restarts, 1);
        assert!((t.sigma - 0.28).abs() < 1e-12);
    }

    #[test]
    fn sixdof_unmask_zeros_locked_slice_and_inflates_sigma() {
        let mut t = Trainer::new(1, true, 0.0);
        t.mean.fill(0.5);
        t.best_weights.fill(0.5);
        t.d.fill(1.0);
        t.inflate_unlocked_axes();
        let mask = plane_lock_weight_mask(8);
        for (i, freeze) in mask.iter().enumerate() {
            if *freeze {
                assert_eq!(t.mean[i], 0.0, "mean {i}");
                assert_eq!(t.best_weights[i], 0.0, "w {i}");
                assert!((t.d[i] - 1.25).abs() < 1e-12, "d {i}");
            } else {
                assert!((t.mean[i] - 0.5).abs() < 1e-12, "keep mean {i}");
                assert!((t.d[i] - 1.0).abs() < 1e-12, "keep d {i}");
            }
        }
    }
}
