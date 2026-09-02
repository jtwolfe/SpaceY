//! Nominal trajectory tracker, residual policy, corridor, and fuel bound.

use crate::constants::*;
use crate::earth::{ecef_to_enu, enu_basis, pad_ecef, pad_geodetic};
use crate::math::{clamp, saturate, Vec3};
use crate::vehicle::DestroyReason;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Exo,
    Entry,
    Glide,
    Landing,
}

impl Phase {
    pub fn as_str(self) -> &'static str {
        match self {
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
}

pub fn classify_phase(nav: &Nav, landing_latched: bool) -> Phase {
    if landing_latched {
        return Phase::Landing;
    }
    if should_start_landing(nav) {
        return Phase::Landing;
    }
    if nav.alt > 78_000.0 && nav.q < 80.0 {
        return Phase::Exo;
    }
    if nav.alt > 12_000.0 && nav.speed > v_ref(nav.alt) + 30.0 {
        return Phase::Entry;
    }
    Phase::Glide
}

pub fn should_start_landing(nav: &Nav) -> bool {
    if nav.alt > 12_000.0 && nav.range_h > 4_000.0 {
        return false;
    }
    let t_sl = MERLIN_THRUST_SL_N * N_ENGINES_LANDING as f64;
    let a_up = (t_sl * 0.85 / nav.mass - G0).max(2.0);
    let v_down = (-nav.v_enu.z).max(0.0);
    let s_burn = v_down * v_down / (2.0 * a_up) + 40.0;
    nav.engine_alt < s_burn || (nav.engine_alt < 1_800.0 && v_down > 60.0 && nav.range_h < 3_000.0)
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

pub fn nominal_controls(nav: &Nav, phase: Phase) -> (Controls, Vec3) {
    // Desired body +X (interstage / "up" of the stage).
    let mut desired_x = nav.up;
    let mut u = Controls::default();

    match phase {
        Phase::Exo | Phase::Entry => {
            if nav.v_enu.norm() > 10.0 {
                desired_x = -enu_to_approx(nav.v_enu.normalized(), nav);
            }
            if phase == Phase::Entry {
                let pred = predicted_glide_range(nav);
                let overshoot = pred - nav.range_h;
                let to_pad = Vec3::new(-nav.pos_enu.x, -nav.pos_enu.y, 0.0);
                let closing = to_pad.norm() < 1.0
                    || nav.v_enu.x * to_pad.x + nav.v_enu.y * to_pad.y > 0.0;
                let q_hot = nav.q > 28_000.0 && nav.speed > 480.0;
                let hypersonic = nav.speed > 1_550.0;
                let long = closing && overshoot > 6_000.0 && nav.speed > 500.0;
                if hypersonic || q_hot || long {
                    let need = if hypersonic {
                        nav.speed - 1_400.0
                    } else if q_hot {
                        nav.speed - 380.0
                    } else {
                        overshoot / 12.0
                    };
                    u.n_engines = N_ENGINES_ENTRY;
                    u.throttle = saturate(0.40 + need / 500.0);
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

    (u, desired_x)
}

fn enu_to_approx(v_enu: Vec3, nav: &Nav) -> Vec3 {
    nav.east * v_enu.x + nav.north * v_enu.y + nav.up * v_enu.z
}

fn predicted_glide_range(nav: &Nav) -> f64 {
    let v_h = (nav.v_enu.x * nav.v_enu.x + nav.v_enu.y * nav.v_enu.y).sqrt();
    let v_d = (-nav.v_enu.z).max(20.0);
    // Very rough: current horiz speed × time-to-ground, with drag bleed.
    let t = nav.alt / v_d;
    v_h * t * 0.55
}

pub fn attitude_command(
    body_x: Vec3,
    body_y: Vec3,
    body_z: Vec3,
    omega: Vec3,
    desired_x: Vec3,
    phase: Phase,
) -> (f64, f64, f64, f64, f64) {
    // Rotate +X toward desired +X. Axis = body_x × desired (world), then body.
    let err = body_x.cross(desired_x.normalized());
    let err_body = Vec3::new(err.dot(body_x), err.dot(body_y), err.dot(body_z));
    // Slow rate command (≤ ~8 deg/s) so TVC cannot pump a tumble.
    let wmax = match phase {
        Phase::Landing => 0.35,
        _ => 0.12,
    };
    let w_cmd_y = clamp(1.8 * err_body.y, -wmax, wmax);
    let w_cmd_z = clamp(1.8 * err_body.z, -wmax, wmax);
    let ey = w_cmd_y - omega.y;
    let ez = w_cmd_z - omega.z;
    let fin_pitch = clamp(ey * 2.2, -FIN_MAX_DEFLECT_RAD, FIN_MAX_DEFLECT_RAD);
    let fin_yaw = clamp(ez * 2.2, -FIN_MAX_DEFLECT_RAD, FIN_MAX_DEFLECT_RAD);
    let fin_roll = clamp(-omega.x * 2.0, -FIN_MAX_DEFLECT_RAD, FIN_MAX_DEFLECT_RAD);
    let gmax = if phase == Phase::Landing {
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
        Phase::Entry => 0.20,
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
    350.0 + 18_000.0 * saturate(alt / 80_000.0)
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
    if phase == Phase::Exo || phase == Phase::Entry {
        return false;
    }
    if nav.fuel < 80.0 && nav.engine_alt > 80.0 && nav.speed > 40.0 {
        return true;
    }
    let dv_fuel = MERLIN_ISP_SL_S * G0 * (nav.mass / DRY_MASS_KG.max(1.0)).ln();
    if phase == Phase::Landing {
        let v_down = (-nav.v_enu.z).max(0.0);
        let v_h = (nav.v_enu.x * nav.v_enu.x + nav.v_enu.y * nav.v_enu.y).sqrt();
        return dv_fuel < (v_down + v_h) * 0.45;
    }
    // Only trip in the lower atmosphere, and credit aero braking generously.
    if nav.alt > 18_000.0 {
        return false;
    }
    let t_go = (nav.alt / (-nav.v_enu.z).max(30.0)).min(120.0);
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
    }
}

pub fn destroy_term(reason: DestroyReason) -> TermReason {
    if reason == DestroyReason::None {
        TermReason::None
    } else {
        TermReason::Destroyed
    }
}
