//! Nominal trajectory tracker, residual policy, corridor, and fuel bound.

use crate::constants::*;
use crate::earth::{ecef_to_enu, enu_basis, pad_ecef, pad_geodetic};
use crate::math::{clamp, saturate, Vec3};
use crate::scenario::Scenario;
use crate::vehicle::DestroyReason;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Deorbit,
    Exo,
    Entry,
    Glide,
    Landing,
}

impl Phase {
    pub fn as_str(self) -> &'static str {
        match self {
            Phase::Deorbit => "DEORBIT",
            Phase::Exo => "EXO",
            Phase::Entry => "ENTRY",
            Phase::Glide => "GLIDE",
            Phase::Landing => "LANDING",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TermReason {
    None,
    Success,
    Corridor,
    FuelInfeasible,
    Destroyed,
    Timeout,
    GroundMiss,
}

impl TermReason {
    pub fn as_str(self) -> &'static str {
        match self {
            TermReason::None => "",
            TermReason::Success => "landed",
            TermReason::Corridor => "left landing corridor",
            TermReason::FuelInfeasible => "fuel cannot reach the pad",
            TermReason::Destroyed => "vehicle destroyed",
            TermReason::Timeout => "episode timeout",
            TermReason::GroundMiss => "ground contact off-pad",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Controls {
    pub throttle: f64,
    pub gimbal_y: f64,
    pub gimbal_z: f64,
    pub fin_pitch: f64,
    pub fin_yaw: f64,
    pub fin_roll: f64,
    pub n_engines: u8,
}

impl Default for Controls {
    fn default() -> Self {
        Self {
            throttle: 0.0,
            gimbal_y: 0.0,
            gimbal_z: 0.0,
            fin_pitch: 0.0,
            fin_yaw: 0.0,
            fin_roll: 0.0,
            n_engines: 0,
        }
    }
}

/// Residual linear policy: 6 outputs × (16 features + bias) = 102 weights.
pub const N_FEATURES: usize = 16;
pub const N_ACTIONS: usize = 6;
pub const N_WEIGHTS: usize = N_ACTIONS * (N_FEATURES + 1);

pub fn policy_residual(w: &[f64], feat: &[f64; N_FEATURES]) -> [f64; N_ACTIONS] {
    let mut out = [0.0; N_ACTIONS];
    if w.len() < N_WEIGHTS {
        return out;
    }
    for a in 0..N_ACTIONS {
        let base = a * (N_FEATURES + 1);
        let mut s = w[base + N_FEATURES];
        for i in 0..N_FEATURES {
            s += w[base + i] * feat[i];
        }
        out[a] = s.tanh();
    }
    out
}

#[derive(Clone, Copy, Debug)]
pub struct Nav {
    pub alt: f64,
    pub speed: f64,
    pub v_enu: Vec3,
    pub pos_enu: Vec3,
    pub range_h: f64,
    pub range_gc: f64,
    pub up: Vec3,
    pub east: Vec3,
    pub north: Vec3,
    pub body_x_ecef: Vec3,
    pub tilt: f64,
    pub q: f64,
    pub aoa: f64,
    pub mach: f64,
    pub fuel: f64,
    pub mass: f64,
    pub engine_alt: f64,
    pub periapsis_alt: f64,
}

pub fn classify_phase(
    nav: &Nav,
    landing_latched: bool,
    deorbit_done: bool,
    entry_latched: bool,
    scenario: Scenario,
) -> Phase {
    if landing_latched {
        return Phase::Landing;
    }
    if should_start_landing_for(nav, scenario) {
        return Phase::Landing;
    }
    if !deorbit_done
        && nav.speed > 5_500.0
        && nav.alt > 110_000.0
        && nav.periapsis_alt > DEORBIT_PERI_DONE_M
    {
        return Phase::Deorbit;
    }
    if scenario.is_orbital() {
        // Pad-ENU range_h collapses near the antipode, so theater is
        // great-circle only. Do not open Entry at first Q — that captures
        // 6 000 km uprange.
        let theater = nav.range_gc < LEO_ENTRY_RANGE_M;
        if entry_latched && nav.speed > 900.0 && nav.alt > 8_000.0 {
            return Phase::Entry;
        }
        if theater && deorbit_done && nav.speed > 1_600.0 && nav.alt < 120_000.0 {
            return Phase::Entry;
        }
        // Survive a deep skip far from the pad, but ignore the 200 Pa
        // thermosphere breeze that used to trip the RTLS v_ref catch-all.
        if nav.q > 25_000.0 && nav.speed > 2_000.0 && nav.alt < 80_000.0 {
            return Phase::Entry;
        }
        if !theater && nav.alt > 35_000.0 {
            return Phase::Exo;
        }
        if nav.alt > 12_000.0 && nav.speed > v_ref(nav.alt) + 30.0 {
            return Phase::Entry;
        }
        return Phase::Glide;
    }
    if nav.alt > 78_000.0 && nav.q < 80.0 {
        return Phase::Exo;
    }
    if nav.speed > 3_000.0 && nav.q > 40.0 && nav.alt < 105_000.0 {
        return Phase::Entry;
    }
    if nav.alt > 12_000.0 && nav.speed > v_ref(nav.alt) + 30.0 {
        return Phase::Entry;
    }
    Phase::Glide
}

pub fn should_start_landing(nav: &Nav) -> bool {
    should_start_landing_for(nav, Scenario::Rtls)
}

pub fn should_start_landing_for(nav: &Nav, scenario: Scenario) -> bool {
    let theater = if scenario.is_orbital() {
        LEO_LANDING_THEATER_M
    } else {
        4_000.0
    };
    let range = if scenario.is_orbital() {
        nav.range_gc.min(nav.range_h)
    } else {
        nav.range_h
    };
    if scenario.is_orbital() {
        // Overflight at ~9 km / ~10 km range / ~500 m/s is the landing
        // window. A suicide-burn latch at 1.6 km is already 30 km east.
        // Do not commit at 13 km / 1 km/s — that is a tank dump.
        if range < LEO_LANDING_COMMIT_RANGE_M
            && nav.alt < LEO_LANDING_COMMIT_ALT_M
            && nav.speed > 70.0
            && nav.speed < LEO_LANDING_COMMIT_SPEED_MPS
        {
            return true;
        }
        if range < 8_000.0 && nav.alt < 5_000.0 && nav.speed > 50.0 && nav.speed < 300.0 {
            return true;
        }
    }
    if nav.alt > 12_000.0 && range > theater {
        return false;
    }
    // A hover-slam 500 km downrange is not a landing attempt.
    if scenario.is_orbital() && range > LEO_LANDING_THEATER_M {
        return false;
    }
    if scenario.is_orbital() && range > 25_000.0 && nav.engine_alt > 8_000.0 {
        return false;
    }
    let t_sl = MERLIN_THRUST_SL_N * N_ENGINES_LANDING as f64;
    let a_up = (t_sl * 0.85 / nav.mass - G0).max(2.0);
    let v_down = (-nav.v_enu.z).max(0.0);
    let s_burn = v_down * v_down / (2.0 * a_up) + 40.0;
    let close = if scenario.is_orbital() { 20_000.0 } else { 3_000.0 };
    nav.engine_alt < s_burn
        || (nav.engine_alt < 2_400.0 && v_down > 40.0 && range < close)
}

/// Reference airspeed vs altitude after a successful entry (m/s).
pub fn v_ref(alt: f64) -> f64 {
    if alt > 70_000.0 {
        1_800.0
    } else if alt > 45_000.0 {
        550.0 + 1_000.0 * (alt - 45_000.0) / 25_000.0
    } else if alt > 20_000.0 {
        280.0 + 270.0 * (alt - 20_000.0) / 25_000.0
    } else if alt > 8_000.0 {
        120.0 + 160.0 * (alt - 8_000.0) / 12_000.0
    } else {
        50.0 + 70.0 * alt / 8_000.0
    }
}

pub fn nominal_controls(nav: &Nav, phase: Phase, scenario: Scenario) -> (Controls, Vec3) {
    // Desired body +X (interstage / "up" of the stage).
    let mut desired_x = nav.up;
    let mut u = Controls::default();
    let orbital = scenario.is_orbital();

    match phase {
        Phase::Deorbit => {
            if nav.v_enu.norm() > 10.0 {
                desired_x = -enu_to_approx(nav.v_enu.normalized(), nav);
            }
            let need = (nav.periapsis_alt - DEORBIT_PERI_TARGET_M).max(0.0);
            if need < 2_000.0 {
                u.n_engines = 0;
                u.throttle = 0.0;
            } else {
                // ~50 m/s deorbit — one Merlin, not a 3-engine slam that
                // overshoots the periapsis target in a single tick.
                u.n_engines = N_ENGINES_LANDING;
                u.throttle = saturate(0.55 + need / 120_000.0);
            }
        }
        Phase::Exo | Phase::Entry => {
            let to_pad = Vec3::new(-nav.pos_enu.x, -nav.pos_enu.y, 0.0);
            let closing = to_pad.norm() < 1.0
                || nav.v_enu.x * to_pad.x + nav.v_enu.y * to_pad.y > 0.0;
            let pred = predicted_landing_range(nav);
            let overshoot = if closing {
                pred - nav.range_gc.min(nav.range_h)
            } else {
                pred + nav.range_gc.min(nav.range_h)
            };
            if nav.v_enu.norm() > 10.0 {
                // Pure retrograde far out. In the pad theater a small
                // dive dumps the 8 km overflight so the landing burn
                // is not a 10 km hover from a 300 m/s skip.
                let mut vaim = nav.v_enu.normalized();
                if orbital && nav.v_enu.norm() > 10.0 {
                    // Dive while fins still work (Q ≲ 40 kPa) so the
                    // skip hits 3–5 km at LZ-1, not a flat 8 km overflight.
                    let pre_pulse = nav.alt < 55_000.0
                        && nav.alt > 32_000.0
                        && nav.range_gc < 400_000.0
                        && nav.q < 45_000.0;
                    let theater = nav.range_gc < 100_000.0
                        && nav.alt < 26_000.0
                        && nav.speed > 400.0;
                    let dive = if pre_pulse {
                        0.34
                    } else if theater && nav.alt > 3_200.0 && nav.range_gc < 40_000.0 {
                        0.48
                    } else if theater {
                        0.26
                    } else {
                        0.0
                    };
                    if dive > 0.0 {
                        vaim = Vec3::new(
                            vaim.x,
                            vaim.y,
                            (vaim.z - dive).clamp(-0.98, -0.05),
                        )
                        .normalized();
                    }
                }
                desired_x = -enu_to_approx(vaim, nav);
            }
            if phase == Phase::Entry {
                let q_hot = nav.q > 28_000.0 && nav.speed > 480.0;
                let hypersonic = nav.speed > 1_550.0;
                let long = closing && overshoot > 6_000.0 && nav.speed > 500.0;
                if orbital {
                    leo_entry_burn(nav, closing, overshoot, &mut u);
                } else {
                    let fuel_ok = nav.fuel > 200.0;
                    if fuel_ok && (hypersonic || q_hot || long) {
                        let need = if hypersonic {
                            nav.speed - 1_400.0
                        } else if q_hot {
                            nav.speed - 380.0
                        } else {
                            overshoot / 12.0
                        };
                        u.n_engines = N_ENGINES_ENTRY;
                        let mut thr = saturate(0.40 + need / 500.0);
                        let q_g = (nav.q * REF_AREA_M2 * 1.15 / nav.mass) / G0;
                        if q_g > 6.3 {
                            thr *= (9.0 / q_g.max(1.0)).clamp(0.40, 1.0);
                        }
                        u.throttle = saturate(thr);
                    }
                }
            }
        }
        Phase::Glide => {
            // Tail-first along air-relative velocity, with a small energy-management
            // pitch offset so a long trajectory dives and a short one lofts.
            let to_pad_h = Vec3::new(-nav.pos_enu.x, -nav.pos_enu.y, 0.0);
            let range = to_pad_h.norm().max(1.0);
            let pred = predicted_glide_range(nav);
            let energy_err = (pred - range) / 8_000.0;
            // LEO pad theater: dive from further out so the skip is at
            // 3–4 km over LZ-1, not a flat 8 km overflight.
            let dive = if orbital && nav.range_gc < 70_000.0 && nav.alt < 22_000.0 {
                // The glide predictor under-counts low-altitude drag and
                // calls this "short", which lofts a 6 km / 300 m/s
                // overflight. Dive while we are still high in the
                // last 30 km; loft only once the pad is actually close
                // and we are below ~3.2 km.
                if nav.alt > 3_200.0 && nav.range_gc < 32_000.0 {
                    -0.38
                } else {
                    clamp(-energy_err, -0.55, 0.10)
                }
            } else {
                clamp(-energy_err, -0.20, 0.20)
            };
            let vdir = if nav.v_enu.norm() > 5.0 {
                nav.v_enu.normalized()
            } else {
                Vec3::new(-nav.pos_enu.x / range, -nav.pos_enu.y / range, -0.4).normalized()
            };
            let aim = Vec3::new(vdir.x, vdir.y, (vdir.z + dive).clamp(-0.98, -0.12)).normalized();
            desired_x = -enu_to_approx(aim, nav);
            u.n_engines = 0;
        }
        Phase::Landing => {
            if orbital {
                leo_landing_burn(nav, &mut u, &mut desired_x);
            } else {
                rtls_hover_slam(nav, &mut u, &mut desired_x);
            }
        }
    }

    (u, desired_x)
}

fn enu_to_approx(v_enu: Vec3, nav: &Nav) -> Vec3 {
    nav.east * v_enu.x + nav.north * v_enu.y + nav.up * v_enu.z
}

fn predicted_glide_range(nav: &Nav) -> f64 {
    predicted_landing_range(nav)
}

fn rtls_hover_slam(nav: &Nav, u: &mut Controls, desired_x: &mut Vec3) {
    leo_or_rtls_hover_slam(nav, u, desired_x, false);
}

/// Hover-slam: track v_des(h) = −√(2 a h). Do NOT PD on kilometres
/// of altitude — that commanded throttle=0 from 4 km (debug_ep).
fn leo_or_rtls_hover_slam(nav: &Nav, u: &mut Controls, desired_x: &mut Vec3, chase_pad: bool) {
    let pz = nav.engine_alt.max(0.5);
    let vz = nav.v_enu.z;
    let a_land = 8.0;
    let v_des = -((2.0 * a_land * (pz - 4.0).max(0.0)).sqrt()).min(120.0);
    let az_cmd = 2.2 * (v_des - vz) + G0 + if pz < 40.0 { 0.4 * (6.0 - pz) } else { 0.0 };
    let t_max = MERLIN_THRUST_SL_N * N_ENGINES_LANDING as f64;
    u.n_engines = N_ENGINES_LANDING;
    if !chase_pad && (-vz > 80.0 || az_cmd * nav.mass > t_max * 0.90) {
        u.n_engines = N_ENGINES_ENTRY;
    }
    if chase_pad && pz < 220.0 && -vz > 45.0 {
        u.n_engines = N_ENGINES_ENTRY;
    }
    let t_avail = MERLIN_THRUST_SL_N * u.n_engines as f64;
    u.throttle = saturate(az_cmd.max(0.0) * nav.mass / t_avail.max(1.0));
    let vh_now = (nav.v_enu.x * nav.v_enu.x + nav.v_enu.y * nav.v_enu.y).sqrt();
    if chase_pad && (vh_now > 15.0 || nav.range_h > 40.0) && nav.fuel > 200.0 {
        u.throttle = u.throttle.max(if vh_now > 30.0 { 0.70 } else { 0.45 });
        if u.n_engines == 0 {
            u.n_engines = N_ENGINES_LANDING;
        }
    }

    let kp = if chase_pad || pz < 300.0 { 0.32 } else { 0.08 };
    let kd = if chase_pad || pz < 300.0 { 0.90 } else { 0.40 };
    // RTLS: don't chase a pad that is tens of km away. LEO final:
    // the pad is in play — full divert authority.
    let reach = if chase_pad {
        1.0
    } else if nav.range_h > 2_000.0 {
        0.15
    } else {
        1.0
    };
    let ax = (-kp * nav.pos_enu.x - kd * nav.v_enu.x) * reach;
    let ay = (-kp * nav.pos_enu.y - kd * nav.v_enu.y) * reach;
    let horiz = (ax * ax + ay * ay).sqrt();
    let max_tilt = if pz < 60.0 {
        0.10
    } else if chase_pad && pz > 80.0 {
        0.48
    } else {
        0.32
    };
    let tilt = (horiz / az_cmd.max(2.0)).min(max_tilt);
    if horiz > 1e-4 {
        let hdir = (nav.east * ax + nav.north * ay).normalized();
        *desired_x = (nav.up * tilt.cos() + hdir * tilt.sin()).normalized();
    } else {
        *desired_x = nav.up;
    }
}

/// 1-engine retro+down. `desired_x = −aim` so a *positive* aim.z
/// pitches the engines down (the old −0.12 "dive" lofted). Only in
/// the last few km — lighting from 18 km spent the stash and stopped
/// 8 km short of a landable state.
fn leo_approach_brake(nav: &Nav, u: &mut Controls, desired_x: &mut Vec3) {
    let pz = nav.engine_alt.max(0.5);
    let vh = (nav.v_enu.x * nav.v_enu.x + nav.v_enu.y * nav.v_enu.y).sqrt();
    let range = nav.range_gc.min(nav.range_h);
    let to_pad_early = Vec3::new(-nav.pos_enu.x, -nav.pos_enu.y, 0.0);
    let inbound_early = to_pad_early.norm() < 1.0
        || nav.v_enu.x * to_pad_early.x + nav.v_enu.y * to_pad_early.y > 0.0;
    let outbound_early = !inbound_early && vh > 18.0;
    let high_over_pad = range < 2_200.0 && pz > 1_800.0 && pz < 7_500.0;
    if nav.fuel < 480.0 {
        return;
    }
    if !outbound_early && !high_over_pad && (pz > 10_000.0 || range > 12_000.0) {
        return;
    }
    if !outbound_early && (nav.q > 80_000.0 || nav.speed > 280.0) && !high_over_pad {
        return;
    }
    // Do not chase a 20 km east slide — that is the #4 miss. Once we
    // are outbound and more than 6 km past LZ-1, fall and suicide.
    if outbound_early && (pz > 10_000.0 || range > 6_000.0 || nav.speed > 600.0) {
        return;
    }
    // Pure retrograde is AoA≈0 and is legal at any Q. A down-pitch
    // at 15 kPa is what broke the airframe on the first dive-brake.
    let vh_tgt = leo_vh_target(range, pz);
    let to_pad = Vec3::new(-nav.pos_enu.x, -nav.pos_enu.y, 0.0);
    let inbound = to_pad.norm() < 1.0
        || nav.v_enu.x * to_pad.x + nav.v_enu.y * to_pad.y > 0.0;
    let outbound = !inbound && vh > 18.0;
    let high_over_pad = range < 2_200.0 && pz > 2_400.0;
    if vh < vh_tgt + 15.0 && !outbound && !high_over_pad {
        return;
    }
    // Always track air-relative velocity when Q can trip the 42° AoA
    // limit. Horizontal-only aim while falling at 40° is a breakup.
    let aim = if nav.v_enu.norm() > 6.0 {
        nav.v_enu.normalized()
    } else if to_pad.norm() > 1.0 {
        Vec3::new(-to_pad.x, -to_pad.y, -0.2).normalized()
    } else {
        Vec3::new(0.0, 0.0, -1.0)
    };
    *desired_x = -enu_to_approx(aim, nav);
    u.n_engines = N_ENGINES_LANDING;
    u.throttle = if high_over_pad {
        saturate(0.70)
    } else {
        saturate(0.50 + (vh - vh_tgt).max(0.0) / 480.0)
    };
}

fn leo_vh_target(range: f64, pz: f64) -> f64 {
    let _ = pz;
    (35.0 + range * 0.016).clamp(20.0, 180.0)
}

/// LEO pad-theater landing. #4 (and the pre-fix intercept) reached
/// ~1 km / ~100 m/s / <1 km / ~2 t, then either hovered (T/W_min > 1
/// on a light 1-engine stack) or chased a walking ECEF pad. Sequence:
/// 1-engine suicide along −velocity, throttled so remaining distance
/// kills leftover speed at ~8 m engine alt. T/W_min > 1 — cut the
/// instant vz > 0. 3-engine only when 1-engine cannot stop in time.
fn leo_landing_burn(nav: &Nav, u: &mut Controls, desired_x: &mut Vec3) {
    let pz = nav.engine_alt.max(0.5);
    let vz = nav.v_enu.z;
    let vh = (nav.v_enu.x * nav.v_enu.x + nav.v_enu.y * nav.v_enu.y).sqrt();
    let range = nav.range_gc.min(nav.range_h);
    let to_pad = Vec3::new(-nav.pos_enu.x, -nav.pos_enu.y, 0.0);
    let inbound = to_pad.norm() < 1.0
        || nav.v_enu.x * to_pad.x + nav.v_enu.y * to_pad.y > 0.0;

    // In / next to the success box: pulse, do not hover-climb.
    if pz < 14.0 && nav.speed < 8.0 && range < 28.0 && nav.fuel > 25.0 {
        leo_terminal_hover(nav, u, desired_x);
        return;
    }

    let a1 = (MERLIN_THRUST_SL_N * 0.90 / nav.mass - G0).max(6.0);
    let v = nav.speed.max(1.0);
    let s1 = v * v / (2.0 * a1) + 12.0;
    let s_box = (pz - 8.0).max(4.0);
    let a_need_1 = (v * v - 25.0).max(0.0) / (2.0 * s_box);

    // Last half-kilometre: time the center engine to ~5 m/s at ~8 m.
    // Full-throttle from 80–250 m empties the tanks at 50–170 m; dropping
    // the burn at 80 kg then falls through the box. Stay on this law
    // down to the last tens of kilograms.
    if pz < 500.0 && range < 3_000.0 && nav.fuel > 25.0 && (v > 5.0 || pz > 14.0) {
        let need_3 = a_need_1 > a1 * 1.08 && pz < 200.0 && v > 50.0 && nav.fuel > 250.0;
        leo_suicide_slam(nav, u, desired_x, to_pad, range, vh, vz, pz, nav.q, need_3);
        return;
    }

    // Cluster only when the center engine is truly too late.
    let need_3 = pz < s1 * 0.50 && pz < 180.0 && v > 80.0 && nav.fuel > 250.0;
    if need_3 {
        leo_suicide_slam(nav, u, desired_x, to_pad, range, vh, vz, pz, nav.q, true);
        return;
    }

    let on_1 = nav.fuel > 180.0
        && pz < s1 * 1.20
        && (pz < 2_400.0 || -vz > 80.0)
        && (range < 12_000.0 || !inbound);
    if on_1 {
        leo_suicide_slam(nav, u, desired_x, to_pad, range, vh, vz, pz, nav.q, false);
        return;
    }

    // Above the curve. Brake leftover horizontal only if a ballistic
    // fall would miss the pad; otherwise fall into the suicide box.
    let vd = (-vz).max(20.0);
    let disc = vd * vd + 2.0 * G0 * pz;
    let t_fall = if disc > 0.0 {
        (-vd + disc.sqrt()) / G0
    } else {
        pz / vd
    };
    let miss = if inbound {
        (range - vh * t_fall).abs()
    } else {
        range + vh * t_fall
    };
    if nav.fuel > 500.0
        && nav.q < 50_000.0
        && pz < 4_500.0
        && nav.speed < 280.0
        && range < 12_000.0
        && (miss > 80.0 && vh > 25.0 || (!inbound && vh > 20.0 && range < 6_000.0))
    {
        leo_approach_brake(nav, u, desired_x);
        if u.n_engines > 0 {
            return;
        }
    }

    let vdir = if nav.v_enu.norm() > 6.0 {
        nav.v_enu.normalized()
    } else {
        Vec3::new(-nav.pos_enu.x, -nav.pos_enu.y, -0.7).normalized()
    };
    *desired_x = -enu_to_approx(vdir, nav);
    u.n_engines = 0;
    u.throttle = 0.0;
}

/// Last tens of metres over LZ-1. A light Merlin cannot hover-descend
/// (T/W_min ≈ 1.25 with ~2 t remaining) — pulse onto the suicide
/// flatten instead of commanding a 70 m/s sink from 1 km.
fn leo_terminal_hover(nav: &Nav, u: &mut Controls, desired_x: &mut Vec3) {
    let pz = nav.engine_alt.max(0.5);
    let vz = nav.v_enu.z;
    let _vh = (nav.v_enu.x * nav.v_enu.x + nav.v_enu.y * nav.v_enu.y).sqrt();
    let v_des = if pz > 40.0 {
        -8.0
    } else if pz > 12.0 {
        -3.0
    } else {
        -1.2
    };
    let az_cmd = 3.0 * (v_des - vz) + G0 + if pz < 12.0 { 0.8 * (8.0 - pz) } else { 0.0 };

    let kp = 0.55;
    let kd = 1.20;
    let ax = -kp * nav.pos_enu.x - kd * nav.v_enu.x;
    let ay = -kp * nav.pos_enu.y - kd * nav.v_enu.y;
    let horiz = (ax * ax + ay * ay).sqrt();

    if vz > 0.12 {
        u.n_engines = 0;
        u.throttle = 0.0;
        *desired_x = nav.up;
        return;
    }

    u.n_engines = N_ENGINES_LANDING;
    let t_avail = MERLIN_THRUST_SL_N * u.n_engines as f64;
    u.throttle = saturate(az_cmd.max(0.0) * nav.mass / t_avail.max(1.0));

    let max_tilt = if pz < 12.0 {
        0.08
    } else if pz < 35.0 {
        0.22
    } else {
        0.38
    };
    let tilt = (horiz / az_cmd.max(4.0)).min(max_tilt);
    if horiz > 1e-4 {
        let hdir = (nav.east * ax + nav.north * ay).normalized();
        *desired_x = (nav.up * tilt.cos() + hdir * tilt.sin()).normalized();
    } else {
        *desired_x = nav.up;
    }
}

fn leo_suicide_slam(
    nav: &Nav,
    u: &mut Controls,
    desired_x: &mut Vec3,
    to_pad: Vec3,
    range: f64,
    _vh: f64,
    vz: f64,
    pz: f64,
    q: f64,
    three: bool,
) {
    let vdir = if nav.v_enu.norm() > 5.0 {
        nav.v_enu.normalized()
    } else {
        Vec3::new(0.0, 0.0, -1.0)
    };
    let mut aim = vdir;
    // Small pad mix only when slow and low. Mixing at 40 m/s / 200 m
    // rotated tilt 10° → 34° and wasted the stash sideways.
    if range > 8.0 && q < 8_000.0 && pz < 90.0 && nav.speed < 22.0 {
        let pad_h = to_pad.normalized();
        let mix = ((range - 8.0) / 50.0).clamp(0.0, 0.22);
        aim = (aim * (1.0 - mix) + Vec3::new(pad_h.x, pad_h.y, 0.0) * mix).normalized();
    }
    *desired_x = -enu_to_approx(aim, nav);
    let up_w = if q > 11_000.0 || pz > 120.0 || nav.speed > 22.0 {
        0.0
    } else if pz < 40.0 && nav.speed < 12.0 {
        0.85
    } else {
        0.20
    };
    *desired_x = (*desired_x * (1.0 - up_w) + nav.up * up_w).normalized();

    // T/W_min > 1: any burn after vz crosses zero climbs out of the box.
    if vz > 0.12 {
        u.n_engines = 0;
        u.throttle = 0.0;
        return;
    }

    u.n_engines = if three {
        N_ENGINES_ENTRY
    } else {
        N_ENGINES_LANDING
    };
    let t_avail = MERLIN_THRUST_SL_N * u.n_engines as f64;
    // Distance-to-go suicide: a_need so we hit ~5 m/s at ~8 m engine alt.
    // Soft enough that 250 m / 60 m/s is ~50% (not a tank dump that
    // stops at 170 m), late enough that 80 m / 34 m/s still lights.
    let s_box = (pz - 8.0).max(4.0);
    let a_need = (nav.speed * nav.speed - 25.0).max(0.0) / (2.0 * s_box);
    let a_hold = if pz < 90.0 && nav.speed > 6.0 {
        // Below the curve, still descending: keep ~constant speed so
        // g does not rebuild a 40 m/s hit from an 85 m / 22 m/s coast.
        0.35
    } else {
        0.0
    };
    let a_cmd = a_need.max(a_hold);
    if a_cmd < 0.4 && !(pz < 20.0 && nav.speed > 5.0) {
        u.n_engines = 0;
        u.throttle = 0.0;
        return;
    }
    let thr = saturate((a_cmd + G0) * nav.mass / t_avail.max(1.0));
    u.throttle = if three && nav.speed > 80.0 {
        1.0
    } else if pz < 20.0 && nav.speed > 5.0 && vz < 0.0 {
        thr.max(THROTTLE_MIN)
    } else {
        thr
    };
}

/// Rough remaining ground range to impact. At 65 km / 4.7 km/s this
/// was ~600 km vs a measured 490 km skip — good enough to trim.
fn predicted_landing_range(nav: &Nav) -> f64 {
    let v_h = (nav.v_enu.x * nav.v_enu.x + nav.v_enu.y * nav.v_enu.y).sqrt();
    let v_d = (-nav.v_enu.z).max(15.0);
    let t = nav.alt / v_d;
    let drag = (1.0 + nav.q / 50_000.0).min(3.5);
    v_h * t * 0.55 / drag
}

/// LEO capture: vacuum slam only while inbound, then a Q-hold / overshoot
/// trim that *protects* the landing reserve. The old `range < 720 km`
/// gate stayed true after overflight and burned the tanks dry 400 km
/// past LZ-1.
fn leo_entry_burn(nav: &Nav, inbound: bool, overshoot: f64, u: &mut Controls) {
    let q = nav.q;
    // Survival burn only if we are about to trip LEO max-Q (250 kPa).
    // Earlier dips into the landing reserve dried the tanks 30 km short.
    let survive = q > 249_000.0;
    let vacuum = inbound
        && nav.range_gc < LEO_CAPTURE_RANGE_M
        && nav.speed > 2_200.0
        && nav.alt > 58_000.0
        && q < 8_000.0;
    let q_hold = q > 55_000.0 && nav.speed > 700.0;
    let trim = inbound && overshoot > 80_000.0 && nav.speed > 1_200.0 && q > 12_000.0;
    // Do not 3-engine in the last 30 km — that is the landing-theater
    // 1-engine brake's job. A cluster pulse at 16 km dumps the stash.

    let reserve = if survive {
        1_500.0
    } else if vacuum {
        LEO_PULSE_RESERVE_KG
    } else {
        // Q-hold / trim may spend down to the landing stash so the
        // 40–20 km pulse actually brakes. The old 8 t floor stopped
        // the burn at 30 km / 4 km/s; they then coasted into 249 kPa
        // and refused every later burn (fuel already < 8 t).
        LEO_LANDING_STASH_KG
    };
    if nav.fuel <= reserve + 200.0 {
        return;
    }
    if !(survive || vacuum || q_hold || trim) {
        return;
    }

    u.n_engines = N_ENGINES_ENTRY;
    let q_tgt = 90_000.0;
    let q_err = ((q - 0.30 * q_tgt) / q_tgt).max(0.0);
    let v_need = ((nav.speed - 1_200.0) / 2_400.0).clamp(0.0, 1.0);
    let o_need = (overshoot / 280_000.0).clamp(0.0, 1.0);
    let mut thr = if vacuum {
        saturate(0.70 + 0.30 * v_need)
    } else {
        saturate(0.40 + 1.00 * q_err + 0.55 * v_need + 0.50 * o_need)
    };
    let q_g = (q * REF_AREA_M2 * 1.15 / nav.mass) / G0;
    if q_g > 12.0 {
        thr *= (12.0 / q_g).clamp(0.40, 1.0);
    }
    u.throttle = saturate(thr);
}

pub fn attitude_command(
    body_x: Vec3,
    body_y: Vec3,
    body_z: Vec3,
    omega: Vec3,
    desired_x: Vec3,
    phase: Phase,
    q_dyn: f64,
) -> (f64, f64, f64, f64, f64) {
    // Rotate +X toward desired +X. Axis = body_x × desired (world), then body.
    let err = body_x.cross(desired_x.normalized());
    let err_body = Vec3::new(err.dot(body_x), err.dot(body_y), err.dot(body_z));
    // Slow rate command (≤ ~8 deg/s) so TVC cannot pump a tumble.
    let wmax = match phase {
        Phase::Landing if q_dyn < 12_000.0 => 0.35,
        Phase::Glide if q_dyn < 110_000.0 => 0.32,
        Phase::Landing if q_dyn < 40_000.0 => 0.28,
        _ => 0.12,
    };
    let w_cmd_y = clamp(1.8 * err_body.y, -wmax, wmax);
    let w_cmd_z = clamp(1.8 * err_body.z, -wmax, wmax);
    let ey = w_cmd_y - omega.y;
    let ez = w_cmd_z - omega.z;
    // Fin moment ~ q S Cl δ. Hold deflection down as Q rises or the
    // loop rate-saturates (structural spin at ~35 km / 100 kPa).
    // Hypersonic entry keeps a tight fin cap (q/2500) so 100 kPa does
    // not rate-saturate. Pad-theater glide/landing was left with ~1°
    // at 40 kPa — weathercock then locked a 10° path and the skip
    // overflew LZ-1 at 7 km. Give the grids enough δ to hold a dive.
    let qn = match phase {
        Phase::Glide | Phase::Landing if q_dyn < 110_000.0 => {
            (1.0 + q_dyn / 90_000.0).max(1.0)
        }
        _ => (1.0 + q_dyn / 2_500.0).max(1.0),
    };
    let fin_lim = FIN_MAX_DEFLECT_RAD / qn.max(1.0);
    // RCS owns the rarefied band; fins at 1–8 kPa with a 70 ms step PIO.
    let fin_enable = if q_dyn < 4_000.0 { 0.0 } else { 1.0 };
    let fin_pitch = clamp(ey * 2.2 / qn, -fin_lim, fin_lim) * fin_enable;
    let fin_yaw = clamp(ez * 2.2 / qn, -fin_lim, fin_lim) * fin_enable;
    let fin_roll = clamp(-omega.x * 1.4 / qn, -fin_lim, fin_lim) * fin_enable;
    // Landing TVC once Q is moderate. Also allow a little gimbal in
    // the late glide so a 1-engine dive-brake can pitch down.
    let gmax = if (phase == Phase::Landing && q_dyn < 20_000.0)
        || (phase == Phase::Glide && q_dyn < 20_000.0)
    {
        GIMBAL_MAX_RAD
    } else {
        0.0
    };
    // Landing TVC only. Entry-burn gimbal, even at 1°, pumped a tumble
    // (debug_ep); weathercock + fins handle exo/entry/glide.
    let gim_y = clamp(-ey * 0.40, -gmax, gmax);
    let gim_z = clamp(-ez * 0.40, -gmax, gmax);
    (gim_y, gim_z, fin_pitch, fin_yaw, fin_roll)
}

pub fn apply_residual(mut u: Controls, r: &[f64; N_ACTIONS], phase: Phase) -> Controls {
    let t_scale = match phase {
        Phase::Landing => 0.25,
        Phase::Entry | Phase::Deorbit => 0.20,
        _ => 0.10,
    };
    u.throttle = saturate(u.throttle + r[0] * t_scale);
    u.gimbal_y = clamp(
        u.gimbal_y + r[1] * GIMBAL_MAX_RAD,
        -GIMBAL_MAX_RAD,
        GIMBAL_MAX_RAD,
    );
    u.gimbal_z = clamp(
        u.gimbal_z + r[2] * GIMBAL_MAX_RAD,
        -GIMBAL_MAX_RAD,
        GIMBAL_MAX_RAD,
    );
    u.fin_pitch = clamp(
        u.fin_pitch + r[3] * FIN_MAX_DEFLECT_RAD,
        -FIN_MAX_DEFLECT_RAD,
        FIN_MAX_DEFLECT_RAD,
    );
    u.fin_yaw = clamp(
        u.fin_yaw + r[4] * FIN_MAX_DEFLECT_RAD,
        -FIN_MAX_DEFLECT_RAD,
        FIN_MAX_DEFLECT_RAD,
    );
    u.fin_roll = clamp(
        u.fin_roll + r[5] * FIN_MAX_DEFLECT_RAD,
        -FIN_MAX_DEFLECT_RAD,
        FIN_MAX_DEFLECT_RAD,
    );
    if u.throttle > 0.05 && u.n_engines == 0 {
        u.n_engines = if phase == Phase::Landing {
            N_ENGINES_LANDING
        } else {
            N_ENGINES_ENTRY
        };
    }
    u
}

pub fn features(nav: &Nav, phase: Phase, pred_overshoot: f64) -> [f64; N_FEATURES] {
    [
        nav.alt / 80_000.0,
        nav.speed / 2_500.0,
        nav.v_enu.z / 400.0,
        nav.range_h / 100_000.0,
        nav.pos_enu.y / 8_000.0,
        nav.pos_enu.x / 8_000.0,
        nav.v_enu.x / 200.0,
        nav.v_enu.y / 200.0,
        nav.aoa / 0.6,
        nav.q / 40_000.0,
        nav.fuel / START_FUEL_KG,
        nav.tilt / 0.8,
        pred_overshoot / 15_000.0,
        nav.mach / 8.0,
        match phase {
            Phase::Deorbit => -0.15,
            Phase::Exo => 0.0,
            Phase::Entry => 0.33,
            Phase::Glide => 0.66,
            Phase::Landing => 1.0,
        },
        nav.engine_alt / 10_000.0,
    ]
}

/// Corridor half-width vs altitude: tightens toward the pad.
/// Measured as **ground-track crossrange** (see `corridor_offset`), not 3D
/// distance to the start→pad chord — a ballistic arc sits far above that chord.
pub fn corridor_radius(alt: f64) -> f64 {
    corridor_radius_for(alt, Scenario::Rtls)
}

pub fn corridor_radius_for(alt: f64, scenario: Scenario) -> f64 {
    match scenario {
        Scenario::Rtls => 350.0 + 18_000.0 * saturate(alt / 80_000.0),
        Scenario::LeoDeorbit => {
            if alt > 65_000.0 {
                f64::INFINITY
            } else {
                LEO_CORRIDOR_PAD_M + 48_000.0 * saturate(alt / 65_000.0)
            }
        }
    }
}

/// LEO pad-ENU crossrange is meaningless on a half-rev coast — at the
/// 100 km interface the pad is still ~6 000 km away and a 56 km corridor
/// instantly kills every episode. Enforce a pad-centered cone only inside
/// the landing theater.
pub fn corridor_violated(nav: &Nav, offset: f64, scenario: Scenario) -> bool {
    match scenario {
        Scenario::Rtls => offset > corridor_radius_for(nav.alt, scenario),
        Scenario::LeoDeorbit => {
            if nav.range_gc > LEO_CORRIDOR_THEATER_M {
                return false;
            }
            let cone = LEO_CORRIDOR_PAD_M + LEO_CORRIDOR_SLOPE * nav.alt.max(0.0);
            nav.range_gc > cone || nav.range_h > cone
        }
    }
}

pub fn corridor_offset(r_ecef: Vec3, start_ecef: Vec3, pad: Vec3) -> f64 {
    let g = pad_geodetic();
    let p = ecef_to_enu(r_ecef, pad, g.lat, g.lon);
    let s = ecef_to_enu(start_ecef, pad, g.lat, g.lon);
    let ab = Vec3::new(-s.x, -s.y, 0.0);
    let ap = Vec3::new(p.x - s.x, p.y - s.y, 0.0);
    let len2 = ab.norm_squared().max(1.0);
    let t = clamp(ap.dot(ab) / len2, -0.15, 1.15);
    let closest = Vec3::new(s.x + ab.x * t, s.y + ab.y * t, 0.0);
    let d = Vec3::new(p.x, p.y, 0.0) - closest;
    d.norm()
}

/// Conservative propulsive reachability. Aero braking is credited only while
/// dynamic pressure can still do useful work.
pub fn fuel_infeasible(nav: &Nav, phase: Phase) -> bool {
    fuel_infeasible_for(nav, phase, Scenario::Rtls)
}

/// Fuel-to-pad bound. LEO only trips this in/near the landing theater —
/// a 500 km skip that latches "landing" far from LZ-1 is a guidance miss,
/// not an instant dry-tank call once the vehicle is actually over the pad.
pub fn fuel_infeasible_for(nav: &Nav, phase: Phase, scenario: Scenario) -> bool {
    if phase == Phase::Deorbit || phase == Phase::Exo || phase == Phase::Entry {
        return false;
    }
    if nav.fuel < 80.0 && nav.engine_alt > 80.0 && nav.speed > 40.0 {
        return true;
    }
    let dv_fuel = MERLIN_ISP_SL_S * G0 * (nav.mass / DRY_MASS_KG.max(1.0)).ln();
    let v_down = (-nav.v_enu.z).max(0.0);
    let v_h = (nav.v_enu.x * nav.v_enu.x + nav.v_enu.y * nav.v_enu.y).sqrt();
    let range = if scenario.is_orbital() {
        nav.range_gc.min(nav.range_h)
    } else {
        nav.range_h
    };

    if scenario.is_orbital()
        && (phase == Phase::Landing || (nav.alt < 10_000.0 && range < LEO_LANDING_THEATER_M))
    {
        // Honest: a hover-slam cannot translate 80+ km.
        if range > LEO_PAD_REACH_M {
            return true;
        }
        // Above ~3 km, aero is still on the clock. Do not trip the
        // bound on the first landing ticks of a 250 m/s / 8 km
        // overflight — that is a guidance problem, not a dry tank.
        if nav.alt > 1_200.0 && range < 15_000.0 && nav.fuel > 150.0 {
            return false;
        }
        // Last-kilometre suicide: do not trip while a stash remains.
        if range < 2_000.0 && nav.engine_alt < 1_500.0 && nav.fuel > 120.0 {
            return false;
        }
        // Last metres: a 40–80 kg stack at 10–30 m/s is still on the
        // suicide curve. The old `need * 0.40` (+60 m/s pad) killed
        // seed 88 at 50 m / 11 m/s / 80 kg — still in play.
        if nav.engine_alt < 200.0 && range < 120.0 && nav.fuel > 25.0 && nav.speed < 50.0 {
            return false;
        }
        // In theater: only fail if we cannot kill the remaining velocity.
        // Do not trip on the first landing tick when tanks still hold a
        // landing reserve (~8 t → ~750 m/s).
        let need = v_down + 0.40 * v_h + 60.0;
        return dv_fuel < need * 0.40;
    }

    if phase == Phase::Landing {
        return dv_fuel < (v_down + v_h) * 0.45;
    }
    // Only trip in the lower atmosphere, and credit aero braking generously.
    if nav.alt > 18_000.0 {
        return false;
    }
    let t_go = (nav.alt / v_down.max(30.0)).min(120.0);
    let a_aero = (nav.q * REF_AREA_M2 * 0.9 / nav.mass).min(40.0);
    let dv_aero = a_aero * t_go * 0.55;
    let need = (nav.speed - dv_aero).max(0.0) + 0.25 * G0 * t_go;
    dv_fuel + dv_aero < need * 0.50
}

pub fn success(nav: &Nav, intact: bool) -> bool {
    intact
        && nav.engine_alt < SUCCESS_ENGINE_ALT_M
        && nav.speed < SUCCESS_SPEED_MPS
        && {
            let vh = (nav.v_enu.x * nav.v_enu.x + nav.v_enu.y * nav.v_enu.y).sqrt();
            vh < SUCCESS_HVEL_MPS
        }
        && nav.range_h < SUCCESS_PAD_OFFSET_M
        && nav.tilt < SUCCESS_TILT_RAD
}

pub fn ground_hit(nav: &Nav) -> bool {
    nav.engine_alt < 0.4
}

pub fn evaluate_contact(nav: &Nav, intact: bool) -> TermReason {
    if success(nav, intact) {
        TermReason::Success
    } else {
        TermReason::GroundMiss
    }
}

pub fn impact_destroy(nav: &Nav) -> bool {
    nav.speed > 8.0 || nav.tilt > 20.0 * std::f64::consts::PI / 180.0
}

pub fn nav_from(
    r_ecef: Vec3,
    v_ground_ecef: Vec3,
    body_x_ecef: Vec3,
    geo_alt: f64,
    engine_alt: f64,
    q: f64,
    aoa: f64,
    mach: f64,
    fuel: f64,
    mass: f64,
) -> Nav {
    let pad = pad_ecef();
    let g = pad_geodetic();
    let (east, north, up) = enu_basis(g.lat, g.lon);
    let pos_enu = ecef_to_enu(r_ecef, pad, g.lat, g.lon);
    let v_enu = Vec3::new(
        v_ground_ecef.dot(east),
        v_ground_ecef.dot(north),
        v_ground_ecef.dot(up),
    );
    Nav {
        alt: geo_alt,
        speed: v_ground_ecef.norm(),
        v_enu,
        pos_enu,
        range_h: (pos_enu.x * pos_enu.x + pos_enu.y * pos_enu.y).sqrt(),
        range_gc: crate::earth::great_circle_m(r_ecef, pad),
        up,
        east,
        north,
        body_x_ecef,
        tilt: crate::earth::tilt_from_vertical(body_x_ecef, up),
        q,
        aoa,
        mach,
        fuel,
        mass,
        engine_alt,
        periapsis_alt: 0.0,
    }
}


pub fn destroy_term(reason: DestroyReason) -> TermReason {
    if reason == DestroyReason::None {
        TermReason::None
    } else {
        TermReason::Destroyed
    }
}
