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
                // Pure retrograde. A "dive" offset on a tail-first stack
                // points the engines up and lofts the skip (seen at 80→101 km).
                desired_x = -enu_to_approx(nav.v_enu.normalized(), nav);
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
            let dive = clamp(-energy_err, -0.20, 0.20);
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
            let pz = nav.engine_alt.max(0.5);
            let vz = nav.v_enu.z;
            // LEO arrives at the theater still hypersonic-ish. Snapping to a
            // vertical hover-slam at 300 m/s / 20 kPa is a 60° AoA death.
            // Stay tail-first / retrograde until the air is slow or thin.
            let leo_retro = orbital
                && nav.speed > 90.0
                && (nav.q > 8_000.0 || nav.speed > 160.0)
                && pz > 250.0;
            if leo_retro {
                if nav.v_enu.norm() > 10.0 {
                    desired_x = -enu_to_approx(nav.v_enu.normalized(), nav);
                }
                u.n_engines = N_ENGINES_ENTRY;
                u.throttle = saturate(0.55 + (nav.speed - 90.0) / 400.0);
            } else {
                // Hover-slam: track v_des(h) = −√(2 a h). Do NOT PD on kilometres
                // of altitude — that commanded throttle=0 from 4 km (debug_ep).
                let a_land = 8.0;
                let v_des = -((2.0 * a_land * (pz - 4.0).max(0.0)).sqrt()).min(120.0);
                let az_cmd = 2.2 * (v_des - vz) + G0 + if pz < 40.0 { 0.4 * (6.0 - pz) } else { 0.0 };
                let t_max = MERLIN_THRUST_SL_N * N_ENGINES_LANDING as f64;
                u.n_engines = N_ENGINES_LANDING;
                if -vz > 80.0 || az_cmd * nav.mass > t_max * 0.90 {
                    u.n_engines = N_ENGINES_ENTRY;
                }
                let t_avail = MERLIN_THRUST_SL_N * u.n_engines as f64;
                u.throttle = saturate(az_cmd.max(0.0) * nav.mass / t_avail.max(1.0));

                let kp = if pz < 300.0 { 0.28 } else { 0.08 };
                let kd = if pz < 300.0 { 0.75 } else { 0.40 };
                // Don't chase a pad that is tens of km away — kill horizontal
                // velocity first; divert only when the pad is in play.
                let reach = if nav.range_h > 2_000.0 { 0.15 } else { 1.0 };
                let ax = (-kp * nav.pos_enu.x - kd * nav.v_enu.x) * reach;
                let ay = (-kp * nav.pos_enu.y - kd * nav.v_enu.y) * reach;
                let horiz = (ax * ax + ay * ay).sqrt();
                let max_tilt = if pz < 60.0 { 0.10 } else { 0.32 };
                let tilt = (horiz / az_cmd.max(2.0)).min(max_tilt);
                if horiz > 1e-4 {
                    let hdir = (nav.east * ax + nav.north * ay).normalized();
                    desired_x = (nav.up * tilt.cos() + hdir * tilt.sin()).normalized();
                } else {
                    desired_x = nav.up;
                }
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
        && nav.alt > 70_000.0
        && q < 8_000.0;
    let q_hold = q > 55_000.0 && nav.speed > 700.0;
    let trim = inbound && overshoot > 80_000.0 && nav.speed > 1_200.0 && q > 12_000.0;
    let near_pad = inbound && nav.range_gc < 25_000.0 && nav.speed > 700.0;

    let reserve = if survive {
        1_500.0
    } else if vacuum {
        LEO_PULSE_RESERVE_KG
    } else {
        LEO_LANDING_FUEL_KG
    };
    if nav.fuel <= reserve + 200.0 {
        return;
    }
    if !(survive || vacuum || q_hold || trim || near_pad) {
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
        Phase::Landing if q_dyn < 6_000.0 => 0.35,
        _ => 0.12,
    };
    let w_cmd_y = clamp(1.8 * err_body.y, -wmax, wmax);
    let w_cmd_z = clamp(1.8 * err_body.z, -wmax, wmax);
    let ey = w_cmd_y - omega.y;
    let ez = w_cmd_z - omega.z;
    // Fin moment ~ q S Cl δ. Hold deflection down as Q rises or the
    // loop rate-saturates (structural spin at ~35 km / 100 kPa).
    let qn = (1.0 + q_dyn / 2_500.0).max(1.0);
    let fin_lim = FIN_MAX_DEFLECT_RAD / qn.max(1.0);
    // RCS owns the rarefied band; fins at 1–8 kPa with a 70 ms step PIO.
    let fin_enable = if q_dyn < 4_000.0 { 0.0 } else { 1.0 };
    let fin_pitch = clamp(ey * 2.2 / qn, -fin_lim, fin_lim) * fin_enable;
    let fin_yaw = clamp(ez * 2.2 / qn, -fin_lim, fin_lim) * fin_enable;
    let fin_roll = clamp(-omega.x * 1.4 / qn, -fin_lim, fin_lim) * fin_enable;
    // Landing TVC only once Q is low. Gimbal at 15–30 kPa with 250 m/s
    // still on the clock is the same tumble that killed entry burns.
    let gmax = if phase == Phase::Landing && q_dyn < 6_000.0 {
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
