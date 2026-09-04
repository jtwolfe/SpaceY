//! Aerodynamics, propulsion, and structural-load checks.

use crate::atmosphere::Air;
use crate::constants::*;
use crate::math::{angle_between, clamp, cos, sin, sqrt, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct Aero {
    pub force_body: Vec3,
    pub moment_body: Vec3,
    pub mach: f64,
    pub q: f64,
    pub aoa: f64,
    pub cd: f64,
    pub heat: f64,
    pub fin_delta: [f64; 4],
}

/// X-config mixer: four lattices at 45/135/225/315°.
pub fn mix_fins(pitch: f64, yaw: f64, roll: f64) -> [f64; 4] {
    let mut d = [0.0; 4];
    for i in 0..4 {
        let th = std::f64::consts::FRAC_PI_4 + i as f64 * std::f64::consts::FRAC_PI_2;
        d[i] = clamp(
            pitch * cos(th) + yaw * sin(th) + roll,
            -FIN_MAX_DEFLECT_RAD,
            FIN_MAX_DEFLECT_RAD,
        );
    }
    d
}

/// 0 below FIN_Q_FADE_PA, 1 at/above FIN_Q_FULL_PA. Pad q cannot move the stack.
pub fn fin_q_enable(q: f64) -> f64 {
    if q <= FIN_Q_FADE_PA {
        0.0
    } else if q >= FIN_Q_FULL_PA {
        1.0
    } else {
        (q - FIN_Q_FADE_PA) / (FIN_Q_FULL_PA - FIN_Q_FADE_PA)
    }
}

/// Mach-dependent axial drag of a tail-first booster (order-of-magnitude curve,
/// not a CFD table). Transonic bump + hypersonic softening.
pub fn cd0(mach: f64) -> f64 {
    if mach < 0.8 {
        0.82
    } else if mach < 1.25 {
        0.82 + 0.55 * (mach - 0.8) / 0.45
    } else if mach < 5.0 {
        1.37 - 0.42 * (mach - 1.25) / 3.75
    } else {
        0.95
    }
}

pub fn aero(
    v_rel_body: Vec3,
    air: Air,
    fin_pitch: f64,
    fin_yaw: f64,
    fin_roll: f64,
) -> Aero {
    let v = v_rel_body.norm();
    let q = 0.5 * air.density * v * v;
    let mach = if air.speed_of_sound > 1.0 {
        v / air.speed_of_sound
    } else {
        0.0
    };
    if v < 0.5 {
        return Aero {
            force_body: Vec3::ZERO,
            moment_body: Vec3::ZERO,
            mach,
            q,
            aoa: 0.0,
            cd: cd0(mach),
            heat: 0.0,
            fin_delta: [0.0; 4],
        };
    }
    let vhat = v_rel_body / v;
    // Tail-first: engines face the flow, body −X into the wind. AoA = 0 is clean.
    let tail = Vec3::new(-1.0, 0.0, 0.0);
    let aoa = angle_between(tail, vhat);
    let s_aoa = sin(aoa);
    let cl = 0.55 * sin(2.0 * aoa);

    let cd_body = cd0(mach) + 1.8 * s_aoa * s_aoa;
    let f_drag = vhat * (-q * REF_AREA_M2 * cd_body);
    let side = tail.cross(vhat);
    let lift_dir = if side.norm() < 1e-8 {
        Vec3::ZERO
    } else {
        vhat.cross(side.normalized()).normalized()
    };
    let f_lift = lift_dir * (q * REF_AREA_M2 * cl);
    let f_body = f_drag + f_lift;
    let r_body = Vec3::new(BODY_CP_X_M, 0.0, 0.0);

    let fp = clamp(fin_pitch, -FIN_MAX_DEFLECT_RAD, FIN_MAX_DEFLECT_RAD);
    let fy = clamp(fin_yaw, -FIN_MAX_DEFLECT_RAD, FIN_MAX_DEFLECT_RAD);
    let fr = clamp(fin_roll, -FIN_MAX_DEFLECT_RAD, FIN_MAX_DEFLECT_RAD);
    let delta = mix_fins(fp, fy, fr);

    let mut f_fins = Vec3::ZERO;
    let mut m_fins = Vec3::ZERO;
    let mut cd_lat = 0.0;
    for i in 0..4 {
        let th = std::f64::consts::FRAC_PI_4 + i as f64 * std::f64::consts::FRAC_PI_2;
        let cy = cos(th);
        let sy = sin(th);
        let r_fin = Vec3::new(FIN_ARM_M, STAGE_RADIUS_M * cy, STAGE_RADIUS_M * sy);
        let d = delta[i];
        let cd_i = FIN_CD0 + 0.85 * d.abs() + 0.30 * s_aoa * s_aoa;
        cd_lat += cd_i * FIN_AREA_M2 / REF_AREA_M2;
        let f_ax = vhat * (-q * FIN_AREA_M2 * cd_i);
        // Hinge along body X at the rim. +δ → tangent force; lever arm at
        // +FIN_ARM_M is the pitch/yaw command. No one-sided |AoA| moment.
        let tangent = Vec3::new(0.0, -sy, cy);
        let f_cmd = tangent * (q * FIN_AREA_M2 * FIN_CL_DELTA * d);
        let f_i = f_ax + f_cmd;
        f_fins += f_i;
        m_fins += r_fin.cross(f_i);
    }

    let heat = 1.83e-4 * v * v * v * sqrt(air.density.max(0.0));
    let f = f_body + f_fins;
    let axial = (-f.dot(vhat)).max(0.0);
    let cd = if q * REF_AREA_M2 > 1e-6 {
        axial / (q * REF_AREA_M2)
    } else {
        cd_body + cd_lat
    };

    Aero {
        force_body: f,
        moment_body: r_body.cross(f_body) + m_fins,
        mach,
        q,
        aoa,
        cd,
        heat,
        fin_delta: delta,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Burn {
    pub force_body: Vec3,
    pub moment_body: Vec3,
    pub mdot: f64,
    pub thrust: f64,
}

/// `n_engines` in {0,1,3}. Throttle is commanded 0..1; if >0 we floor at THROTTLE_MIN.
pub fn propulsion(
    throttle_cmd: f64,
    gimbal_y: f64,
    gimbal_z: f64,
    n_engines: u8,
    pressure_pa: f64,
    fuel: f64,
) -> Burn {
    if n_engines == 0 || fuel <= 1.0 || throttle_cmd <= 0.02 {
        return Burn {
            force_body: Vec3::ZERO,
            moment_body: Vec3::ZERO,
            mdot: 0.0,
            thrust: 0.0,
        };
    }
    let throttle = if throttle_cmd > 0.02 {
        clamp(throttle_cmd, THROTTLE_MIN, THROTTLE_MAX)
    } else {
        0.0
    };
    let p_frac = clamp(pressure_pa / 101_325.0, 0.0, 1.0);
    let thrust_one = lerp(MERLIN_THRUST_VAC_N, MERLIN_THRUST_SL_N, p_frac);
    let isp = lerp(MERLIN_ISP_VAC_S, MERLIN_ISP_SL_S, p_frac);
    let gy = clamp(gimbal_y, -GIMBAL_MAX_RAD, GIMBAL_MAX_RAD);
    let gz = clamp(gimbal_z, -GIMBAL_MAX_RAD, GIMBAL_MAX_RAD);
    // Thrust is toward +X (interstage / "up" the stack). Exhaust leaves the
    // −X engine end. +gimbal_y adds a +Z component at the engines → +My.
    let dir = Vec3::new(cos(gy) * cos(gz), sin(gz), sin(gy) * cos(gz)).normalized();
    let thrust = thrust_one * n_engines as f64 * throttle;
    let force = dir * thrust;
    let r_eng = Vec3::new(-STAGE_LENGTH_M * 0.5, 0.0, 0.0);
    let mdot = thrust / (isp * G0);
    Burn {
        force_body: force,
        moment_body: r_eng.cross(force),
        mdot,
        thrust,
    }
}

/// Plant-side ignition latch. Commanded 10 Hz on/off is a restart, not a throttle.
#[derive(Clone, Debug)]
pub struct EngineGate {
    pub on: bool,
    pub lights: u32,
    pub relights: u32,
    last_light_t: f64,
    last_shutdown_t: f64,
    hold_throttle: f64,
    hold_n: u8,
}

impl Default for EngineGate {
    fn default() -> Self {
        Self {
            on: false,
            lights: 0,
            relights: 0,
            last_light_t: f64::NEG_INFINITY,
            last_shutdown_t: f64::NEG_INFINITY,
            hold_throttle: 0.0,
            hold_n: 0,
        }
    }
}

impl EngineGate {
    fn want(throttle: f64, n_engines: u8) -> bool {
        n_engines > 0 && throttle > 0.02
    }

    pub fn apply(&mut self, t: f64, throttle: f64, n_engines: u8) -> (f64, u8) {
        let want = Self::want(throttle, n_engines);
        if self.on {
            if want {
                self.hold_throttle = throttle.max(THROTTLE_MIN);
                self.hold_n = n_engines.max(1);
                (self.hold_throttle, self.hold_n)
            } else if t - self.last_light_t < ENGINE_MIN_BURN_S {
                (
                    self.hold_throttle.max(THROTTLE_MIN),
                    self.hold_n.max(1),
                )
            } else {
                self.on = false;
                self.last_shutdown_t = t;
                self.hold_throttle = 0.0;
                self.hold_n = 0;
                (0.0, 0)
            }
        } else if want {
            if self.lights > 0 && t - self.last_shutdown_t < ENGINE_RESTART_DELAY_S {
                (0.0, 0)
            } else {
                self.on = true;
                self.last_light_t = t;
                self.lights += 1;
                if self.lights > 1 {
                    self.relights += 1;
                }
                self.hold_throttle = throttle.max(THROTTLE_MIN);
                self.hold_n = n_engines.max(1);
                (self.hold_throttle, self.hold_n)
            }
        } else {
            (0.0, 0)
        }
    }
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DestroyReason {
    None,
    MaxQ,
    OverG,
    QAlpha,
    HighAoA,
    Spin,
    GroundImpact,
}

pub fn check_destruction(q: f64, aoa: f64, accel_g: f64, rate: f64, enabled: bool) -> DestroyReason {
    check_destruction_limits(
        q,
        aoa,
        accel_g,
        rate,
        enabled,
        Q_DESTROY_PA,
        G_DESTROY,
        Q_ALPHA_DESTROY,
        RATE_DESTROY_RAD_S,
    )
}

pub fn check_destruction_limits(
    q: f64,
    aoa: f64,
    accel_g: f64,
    rate: f64,
    enabled: bool,
    q_lim: f64,
    g_lim: f64,
    q_alpha_lim: f64,
    rate_lim: f64,
) -> DestroyReason {
    if !enabled {
        return DestroyReason::None;
    }
    if q > q_lim {
        return DestroyReason::MaxQ;
    }
    if accel_g > g_lim {
        return DestroyReason::OverG;
    }
    let q_kpa = q / 1000.0;
    let aoa_deg = aoa.abs() * 180.0 / std::f64::consts::PI;
    if q_kpa * aoa_deg > q_alpha_lim {
        return DestroyReason::QAlpha;
    }
    if aoa.abs() > AOA_DESTROY_RAD && q > Q_FOR_AOA_DESTROY_PA {
        return DestroyReason::HighAoA;
    }
    if rate > rate_lim {
        return DestroyReason::Spin;
    }
    DestroyReason::None
}

/// Policy-owned RCS: commanded body accel in [-1, 1], Q-fade, no inner PD.
pub fn rcs_commanded(cmd: Vec3, inertia: Vec3, q: f64) -> Vec3 {
    let blend = (1.0 - q / RCS_Q_HANDOFF_PA).clamp(0.0, 1.0);
    if blend < 1e-4 {
        return Vec3::ZERO;
    }
    Vec3::new(
        inertia.x * RCS_ANG_ACCEL * clamp(cmd.x, -1.0, 1.0),
        inertia.y * RCS_ANG_ACCEL * clamp(cmd.y, -1.0, 1.0),
        inertia.z * RCS_ANG_ACCEL * clamp(cmd.z, -1.0, 1.0),
    ) * blend
}

/// Cold-gas RCS moment in the body frame. Authority fades as dynamic
/// pressure comes up so grid fins own the atmosphere.
pub fn rcs_moment(omega: Vec3, err_body: Vec3, inertia: Vec3, q: f64) -> Vec3 {
    let blend = (1.0 - q / RCS_Q_HANDOFF_PA).clamp(0.0, 1.0);
    if blend < 1e-4 {
        return Vec3::ZERO;
    }
    let wmax = 0.14;
    let w_cmd_y = crate::math::clamp(2.2 * err_body.y, -wmax, wmax);
    let w_cmd_z = crate::math::clamp(2.2 * err_body.z, -wmax, wmax);
    let ay = crate::math::clamp((w_cmd_y - omega.y) * 3.0, -RCS_ANG_ACCEL, RCS_ANG_ACCEL);
    let az = crate::math::clamp((w_cmd_z - omega.z) * 3.0, -RCS_ANG_ACCEL, RCS_ANG_ACCEL);
    let ax = crate::math::clamp(-omega.x * 2.5, -RCS_ANG_ACCEL, RCS_ANG_ACCEL);
    Vec3::new(inertia.x * ax, inertia.y * ay, inertia.z * az) * blend
}

impl DestroyReason {
    pub fn as_str(self) -> &'static str {
        match self {
            DestroyReason::None => "",
            DestroyReason::MaxQ => "structural failure (max-Q)",
            DestroyReason::OverG => "structural failure (over-G)",
            DestroyReason::QAlpha => "aero breakup (q-alpha)",
            DestroyReason::HighAoA => "aero breakup (excessive AoA)",
            DestroyReason::Spin => "structural failure (spin)",
            DestroyReason::GroundImpact => "ground impact",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atmosphere;

    fn air_20km() -> Air {
        atmosphere::lookup(20_000.0)
    }

    fn v_at_aoa(speed: f64, aoa: f64) -> Vec3 {
        Vec3::new(-speed * cos(aoa), 0.0, speed * sin(aoa))
    }

    #[test]
    fn fin_q_enable_is_off_on_the_pad() {
        assert_eq!(fin_q_enable(0.0), 0.0);
        assert_eq!(fin_q_enable(FIN_Q_FADE_PA), 0.0);
        assert!((fin_q_enable(FIN_Q_FULL_PA) - 1.0).abs() < 1e-12);
        assert!((fin_q_enable(0.5 * (FIN_Q_FADE_PA + FIN_Q_FULL_PA)) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn engine_first_weathercock_is_restoring() {
        let air = air_20km();
        let v = 750.0;
        let pos = aero(v_at_aoa(v, 0.25), air, 0.0, 0.0, 0.0);
        let neg = aero(v_at_aoa(v, -0.25), air, 0.0, 0.0, 0.0);
        assert!(pos.aoa > 0.2);
        assert!(
            pos.moment_body.y > 0.0,
            "My should restore +AoA (engines toward flow), got {}",
            pos.moment_body.y
        );
        assert!(
            neg.moment_body.y < 0.0,
            "My should restore −AoA, got {}",
            neg.moment_body.y
        );
        assert!(
            (pos.moment_body.y + neg.moment_body.y).abs() < 0.15 * pos.moment_body.y.abs(),
            "restore should be odd in AoA"
        );
    }

    #[test]
    fn grid_fins_beat_bare_body_at_entry_q() {
        let air = air_20km();
        let a = aero(v_at_aoa(900.0, 0.30), air, 0.0, 0.0, 0.0);
        let iy = inertia_diag(wet_mass(20_000.0)).y;
        let alpha_ddot = a.moment_body.y / iy;
        assert!(
            alpha_ddot > 0.015,
            "weathercock too weak: α̈={alpha_ddot:.4} Iy={iy:.0} My={}",
            a.moment_body.y
        );
    }

    #[test]
    fn modest_fin_command_does_not_overpower_weathercock() {
        let air = air_20km();
        let v = 750.0;
        let aoa = 0.22;
        let restore = aero(v_at_aoa(v, aoa), air, 0.0, 0.0, 0.0).moment_body.y;
        let commanded = aero(v_at_aoa(v, aoa), air, 0.08, 0.0, 0.0).moment_body.y;
        let delta = (commanded - restore).abs();
        assert!(
            delta < restore.abs() * 0.55,
            "4.6° pitch cmd overpowered restore: restore={restore:.0} cmd={commanded:.0}"
        );
    }

    #[test]
    fn deployed_grids_add_axial_drag() {
        let air = air_20km();
        let a = aero(v_at_aoa(750.0, 0.0), air, 0.0, 0.0, 0.0);
        let deflected = aero(v_at_aoa(750.0, 0.0), air, 0.20, 0.0, 0.0);
        assert!(
            deflected.cd > a.cd + 0.02,
            "deflection should add drag: trim={} defl={}",
            a.cd,
            deflected.cd
        );
        assert!(
            a.cd > cd0(a.mach) + 0.15,
            "zero-δ lattices still drag: cd={} cd0={}",
            a.cd,
            cd0(a.mach)
        );
    }

    #[test]
    fn x_mixer_is_odd_in_pitch() {
        let p = mix_fins(0.1, 0.0, 0.0);
        let n = mix_fins(-0.1, 0.0, 0.0);
        for i in 0..4 {
            assert!((p[i] + n[i]).abs() < 1e-12);
        }
        assert!(p.iter().any(|d| d.abs() > 0.05));
    }

    #[test]
    fn ten_hz_chatter_is_one_light() {
        let mut g = EngineGate::default();
        for i in 0..20 {
            let t = i as f64 * 0.1;
            let on = i % 2 == 0;
            let (thr, n) = g.apply(t, if on { 0.70 } else { 0.0 }, if on { 1 } else { 0 });
            assert!(thr >= THROTTLE_MIN - 1e-12);
            assert_eq!(n, 1);
        }
        assert_eq!(g.lights, 1);
        assert_eq!(g.relights, 0);
        assert!(g.on);
    }

    #[test]
    fn restart_waits_out_the_delay() {
        let mut g = EngineGate::default();
        g.apply(0.0, 0.70, 1);
        let (thr, n) = g.apply(ENGINE_MIN_BURN_S + 0.01, 0.0, 0);
        assert_eq!(thr, 0.0);
        assert_eq!(n, 0);
        assert!(!g.on);
        let blocked = g.apply(ENGINE_MIN_BURN_S + 1.0, 1.0, 1);
        assert_eq!(blocked, (0.0, 0));
        assert_eq!(g.relights, 0);
        let t_ok = ENGINE_MIN_BURN_S + ENGINE_RESTART_DELAY_S + 0.05;
        let (thr, n) = g.apply(t_ok, 0.80, 1);
        assert!(thr >= THROTTLE_MIN);
        assert_eq!(n, 1);
        assert_eq!(g.lights, 2);
        assert_eq!(g.relights, 1);
    }

    #[test]
    fn first_light_is_always_allowed() {
        let mut g = EngineGate::default();
        let (thr, n) = g.apply(0.0, 0.50, 1);
        assert!(thr >= THROTTLE_MIN);
        assert_eq!(n, 1);
        assert_eq!(g.lights, 1);
        assert_eq!(g.relights, 0);
    }

    #[test]
    fn cluster_change_while_lit_is_not_a_relight() {
        let mut g = EngineGate::default();
        let (_, n1) = g.apply(0.0, 0.70, 1);
        assert_eq!(n1, 1);
        let (_, n3) = g.apply(0.2, 0.80, 3);
        assert_eq!(n3, 3);
        assert_eq!(g.lights, 1);
        assert_eq!(g.relights, 0);
        assert!(g.on);
    }
}
