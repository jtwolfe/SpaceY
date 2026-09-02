//! Aerodynamics, propulsion, and structural-load checks.

use crate::atmosphere::Air;
use crate::constants::*;
use crate::math::{angle_between, clamp, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct Aero {
    pub force_body: Vec3,
    pub moment_body: Vec3,
    pub mach: f64,
    pub q: f64,
    pub aoa: f64,
    pub cd: f64,
    pub heat: f64,
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
        };
    }
    let vhat = v_rel_body / v;
    // Tail-first: engines face the flow, body -X into the wind. AoA = 0 is clean.
    let tail = Vec3::new(-1.0, 0.0, 0.0);
    let aoa = angle_between(tail, vhat);

    let cd = cd0(mach) + 1.8 * aoa.sin().powi(2);
    let cl = 0.55 * (2.0 * aoa).sin();

    // Drag opposite air-relative velocity; lift in the plane of tail × (tail × v).
    let f_drag = vhat * (-q * REF_AREA_M2 * cd);
    let side = tail.cross(vhat);
    let lift_dir = if side.norm() < 1e-8 {
        Vec3::ZERO
    } else {
        vhat.cross(side.normalized()).normalized()
    };
    let f_lift = lift_dir * (q * REF_AREA_M2 * cl);

    // Combined body+fin CP aft of the CG (toward the interstage / +X) so a
    // tail-first booster weathercocks. Bare body is unstable; Block 5 fins win.
    let r_cp = Vec3::new(8.0, 0.0, 0.0);
    let f_body = f_drag + f_lift;

    // Grid fins: four panels, X-configuration, produce force ~ q S Cl_δ δ
    // plus a restoring term from local alpha. Commands are pitch/yaw/roll mixes.
    let cl_d = 2.2;
    let s = FIN_AREA_M2;
    let arm = FIN_ARM_M;
    let fp = clamp(fin_pitch, -FIN_MAX_DEFLECT_RAD, FIN_MAX_DEFLECT_RAD);
    let fy = clamp(fin_yaw, -FIN_MAX_DEFLECT_RAD, FIN_MAX_DEFLECT_RAD);
    let fr = clamp(fin_roll, -FIN_MAX_DEFLECT_RAD, FIN_MAX_DEFLECT_RAD);
    // Pitch moment about +Y, yaw about +Z, roll about +X.
    let m_fins = Vec3::new(
        q * s * cl_d * fr * STAGE_RADIUS_M * 2.4,
        q * s * cl_d * fp * arm * 2.0 - q * s * 0.8 * aoa * arm * vhat.z.signum().abs(),
        q * s * cl_d * fy * arm * 2.0,
    );
    // Small fin force (side force) so translation is affected, not just torque.
    let f_fins = Vec3::new(0.0, -q * s * cl_d * fy * 0.4, q * s * cl_d * fp * 0.4);

    // Sutton–Graves-ish heating proxy ~ k v³ √ρ — for glow, not TPS sizing.
    let heat = 1.83e-4 * v.powi(3) * air.density.max(0.0).sqrt();

    Aero {
        force_body: f_body + f_fins,
        moment_body: r_cp.cross(f_body) + m_fins,
        mach,
        q,
        aoa,
        cd,
        heat,
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
    let dir = Vec3::new(gy.cos() * gz.cos(), gz.sin(), gy.sin() * gz.cos()).normalized();
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
    if !enabled {
        return DestroyReason::None;
    }
    if q > Q_DESTROY_PA {
        return DestroyReason::MaxQ;
    }
    if accel_g > G_DESTROY {
        return DestroyReason::OverG;
    }
    let q_kpa = q / 1000.0;
    let aoa_deg = aoa.abs() * 180.0 / std::f64::consts::PI;
    if q_kpa * aoa_deg > Q_ALPHA_DESTROY {
        return DestroyReason::QAlpha;
    }
    if aoa.abs() > AOA_DESTROY_RAD && q > Q_FOR_AOA_DESTROY_PA {
        return DestroyReason::HighAoA;
    }
    if rate > RATE_DESTROY_RAD_S {
        return DestroyReason::Spin;
    }
    DestroyReason::None
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
