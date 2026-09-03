//! Nominal RTLS tracker (autopilot demo), corridor, and fuel bound.

use crate::constants::*;
use crate::earth::{ecef_to_enu, enu_basis, pad_ecef, pad_geodetic};
use crate::math::{clamp, cos, ln, saturate, sin, sqrt, Vec3};
use crate::scenario::Scenario;
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
            TermReason::GroundMiss => "outside success box",
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
    _deorbit_done: bool,
    _entry_latched: bool,
    scenario: Scenario,
) -> Phase {
    if scenario.is_terminal_hop() || landing_latched {
        return Phase::Landing;
    }
    if should_start_landing_for(nav, scenario) {
        return Phase::Landing;
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
    if scenario.is_terminal_hop() {
        return true;
    }
    let range = nav.range_h;
    if nav.alt > 12_000.0 && range > 4_000.0 {
        return false;
    }
    let t_sl = MERLIN_THRUST_SL_N * N_ENGINES_LANDING as f64;
    let a_up = (t_sl * 0.85 / nav.mass - G0).max(2.0);
    let v_down = (-nav.v_enu.z).max(0.0);
    let s_burn = v_down * v_down / (2.0 * a_up) + 40.0;
    nav.engine_alt < s_burn
        || (nav.engine_alt < 2_400.0 && v_down > 40.0 && range < 3_000.0)
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

pub fn nominal_controls(nav: &Nav, phase: Phase, _scenario: Scenario) -> (Controls, Vec3) {
    // Desired body +X (interstage / "up" of the stage).
    let mut desired_x = nav.up;
    let mut u = Controls::default();

    match phase {
        Phase::Exo | Phase::Entry => {
            let to_pad = Vec3::new(-nav.pos_enu.x, -nav.pos_enu.y, 0.0);
            let closing = to_pad.norm() < 1.0
                || nav.v_enu.x * to_pad.x + nav.v_enu.y * to_pad.y > 0.0;
            let pred = predicted_landing_range(nav);
            let overshoot = if closing {
                pred - nav.range_h
            } else {
                pred + nav.range_h
            };
            if nav.v_enu.norm() > 10.0 {
                desired_x = -enu_to_approx(nav.v_enu.normalized(), nav);
            }
            if phase == Phase::Entry {
                let q_hot = nav.q > 28_000.0 && nav.speed > 480.0;
                let hypersonic = nav.speed > 1_550.0;
                let long = closing && overshoot > 6_000.0 && nav.speed > 500.0;
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
        Phase::Glide => {
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
            rtls_hover_slam(nav, &mut u, &mut desired_x);
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

/// Hover-slam: track v_des(h) = −√(2 a h). Do NOT PD on kilometres
/// of altitude — that commanded throttle=0 from 4 km (debug_ep).
fn rtls_hover_slam(nav: &Nav, u: &mut Controls, desired_x: &mut Vec3) {
    let pz = nav.engine_alt.max(0.5);
    let vz = nav.v_enu.z;
    let a_land = 8.0;
    let v_des = -(sqrt(2.0 * a_land * (pz - 4.0).max(0.0))).min(120.0);
    let az_cmd = 2.2 * (v_des - vz) + G0 + if pz < 40.0 { 0.4 * (6.0 - pz) } else { 0.0 };
    let t_max = MERLIN_THRUST_SL_N * N_ENGINES_LANDING as f64;
    u.n_engines = N_ENGINES_LANDING;
    if -vz > 80.0 || az_cmd * nav.mass > t_max * 0.90 {
        u.n_engines = N_ENGINES_ENTRY;
    }
    let t_avail = MERLIN_THRUST_SL_N * u.n_engines as f64;
    u.throttle = saturate(az_cmd.max(0.0) * nav.mass / t_avail.max(1.0));

    let kp = if pz < 300.0 { 0.32 } else { 0.08 };
    let kd = if pz < 300.0 { 0.90 } else { 0.40 };
    let reach = if nav.range_h > 2_000.0 { 0.15 } else { 1.0 };
    let ax = (-kp * nav.pos_enu.x - kd * nav.v_enu.x) * reach;
    let ay = (-kp * nav.pos_enu.y - kd * nav.v_enu.y) * reach;
    let horiz = sqrt(ax * ax + ay * ay);
    let max_tilt = if pz < 60.0 { 0.10 } else { 0.32 };
    let tilt = (horiz / az_cmd.max(2.0)).min(max_tilt);
    if horiz > 1e-4 {
        let hdir = (nav.east * ax + nav.north * ay).normalized();
        *desired_x = (nav.up * cos(tilt) + hdir * sin(tilt)).normalized();
    } else {
        *desired_x = nav.up;
    }
}

/// Rough remaining ground range to impact. At 65 km / 4.7 km/s this
/// was ~600 km vs a measured 490 km skip — good enough to trim.
fn predicted_landing_range(nav: &Nav) -> f64 {
    let v_h = sqrt(nav.v_enu.x * nav.v_enu.x + nav.v_enu.y * nav.v_enu.y);
    let v_d = (-nav.v_enu.z).max(15.0);
    let t = nav.alt / v_d;
    let drag = (1.0 + nav.q / 50_000.0).min(3.5);
    v_h * t * 0.55 / drag
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

/// Corridor half-width vs altitude: tightens toward the pad.
/// Measured as **ground-track crossrange** (see `corridor_offset`), not 3D
/// distance to the start→pad chord — a ballistic arc sits far above that chord.
pub fn corridor_radius(alt: f64) -> f64 {
    corridor_radius_for(alt, Scenario::Rtls)
}

pub fn corridor_radius_for(alt: f64, scenario: Scenario) -> f64 {
    match scenario {
        Scenario::Pad | Scenario::Slam | Scenario::Attitude | Scenario::Wind => f64::INFINITY,
        Scenario::Glide => 800.0 + 6_000.0 * saturate(alt / 25_000.0),
        Scenario::Rtls => 350.0 + 18_000.0 * saturate(alt / 80_000.0),
    }
}

pub fn corridor_violated(nav: &Nav, offset: f64, scenario: Scenario) -> bool {
    if scenario.is_terminal_hop() {
        return false;
    }
    offset > corridor_radius_for(nav.alt, scenario)
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

pub fn fuel_infeasible_for(nav: &Nav, phase: Phase, scenario: Scenario) -> bool {
    if scenario.is_terminal_hop() {
        return nav.fuel < 40.0 && nav.engine_alt > 40.0 && nav.speed > 25.0;
    }
    if phase == Phase::Exo || phase == Phase::Entry {
        return false;
    }
    if nav.fuel < 80.0 && nav.engine_alt > 80.0 && nav.speed > 40.0 {
        return true;
    }
    let dv_fuel = MERLIN_ISP_SL_S * G0 * ln(nav.mass / DRY_MASS_KG.max(1.0));
    let v_down = (-nav.v_enu.z).max(0.0);
    let v_h = sqrt(nav.v_enu.x * nav.v_enu.x + nav.v_enu.y * nav.v_enu.y);

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
            let vh = sqrt(nav.v_enu.x * nav.v_enu.x + nav.v_enu.y * nav.v_enu.y);
            vh < SUCCESS_HVEL_MPS
        }
        && nav.range_h < SUCCESS_PAD_OFFSET_M
        && nav.tilt < SUCCESS_TILT_RAD
}

/// Min-throttle hover-slam acceleration (T/W − g). Merlin cannot go below 40%.
pub fn slam_a_eff(mass: f64) -> f64 {
    let tw = THROTTLE_MIN * MERLIN_THRUST_SL_N / mass.max(DRY_MASS_KG);
    (tw - G0).max(0.5)
}

/// |v_down*| along a constant-a_eff slam: v² = 2 a h.
pub fn slam_v_ref(engine_alt: f64, mass: f64) -> f64 {
    sqrt((2.0 * slam_a_eff(mass) * engine_alt.max(0.0)).max(0.0))
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
    nav.speed > IMPACT_SPEED_MPS || nav.tilt > IMPACT_TILT_RAD
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
        range_h: sqrt(pos_enu.x * pos_enu.x + pos_enu.y * pos_enu.y),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario::Scenario;

    #[test]
    fn hops_skip_the_corridor() {
        assert!(corridor_radius_for(80.0, Scenario::Pad).is_infinite());
        assert!(!corridor_violated(
            &Nav {
                alt: 80.0,
                speed: 5.0,
                v_enu: Vec3::ZERO,
                pos_enu: Vec3::new(1_000.0, 0.0, 80.0),
                range_h: 1_000.0,
                range_gc: 1_000.0,
                up: Vec3::Z,
                east: Vec3::X,
                north: Vec3::Y,
                body_x_ecef: Vec3::Z,
                tilt: 0.0,
                q: 0.0,
                aoa: 0.0,
                mach: 0.0,
                fuel: 1_000.0,
                mass: DRY_MASS_KG + 1_000.0,
                engine_alt: 80.0,
                periapsis_alt: 0.0,
            },
            1e9,
            Scenario::Pad
        ));
    }

    #[test]
    fn rtls_corridor_tightens_near_the_pad() {
        let high = corridor_radius_for(80_000.0, Scenario::Rtls);
        let low = corridor_radius_for(0.0, Scenario::Rtls);
        assert!(low < high);
        assert!(low < 1_000.0);
    }

    fn touch(speed: f64, vh: f64, range: f64, tilt_deg: f64, engine_alt: f64) -> Nav {
        let vz = -(speed * speed - vh * vh).max(0.0).sqrt();
        Nav {
            alt: engine_alt + 25.0,
            speed,
            v_enu: Vec3::new(vh, 0.0, vz),
            pos_enu: Vec3::new(range, 0.0, engine_alt),
            range_h: range,
            range_gc: range,
            up: Vec3::Z,
            east: Vec3::X,
            north: Vec3::Y,
            body_x_ecef: Vec3::Z,
            tilt: tilt_deg * std::f64::consts::PI / 180.0,
            q: 0.0,
            aoa: 0.0,
            mach: 0.0,
            fuel: 500.0,
            mass: DRY_MASS_KG + 500.0,
            engine_alt,
            periapsis_alt: 0.0,
        }
    }

    #[test]
    fn in_box_touchdown_is_a_land() {
        let n = touch(6.0, 1.0, 8.0, 4.0, 5.0);
        assert!(success(&n, true));
        assert_eq!(evaluate_contact(&n, true), TermReason::Success);
        assert!(!impact_destroy(&n));
    }

    #[test]
    fn hot_pad_sitdown_is_a_miss_not_a_breakup() {
        let n = touch(12.0, 1.5, 6.0, 5.0, 0.2);
        assert!(!success(&n, true));
        assert!(!impact_destroy(&n));
        assert_eq!(evaluate_contact(&n, true), TermReason::GroundMiss);
    }

    #[test]
    fn slap_is_still_a_breakup() {
        let n = touch(24.0, 2.0, 5.0, 4.0, 0.2);
        assert!(impact_destroy(&n));
    }
}
