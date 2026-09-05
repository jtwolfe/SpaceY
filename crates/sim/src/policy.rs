//! CMA-NeuroES policy: a small tanh MLP that *is* the controller.
//!
//! Observations are pad-frame pose, velocity, body rate, and fuel. Actions
//! are throttle, TVC gimbal, grid fins, engine cluster, and RCS. Zero weights
//! keep engines and RCS off (deadband around tanh 0). No inner attitude PD,
//! no wind/q/Mach/phase inputs, no reference path.

use crate::constants::*;
use crate::guidance::{Controls, Nav};
use crate::math::{asinh, clamp, saturate, tanh, Quat, Vec3};

pub const N_IN: usize = 14;
pub const N_OUT: usize = 10;
pub const HIDDEN_START: usize = 8;
/// Frozen. Panic-growth to 24 is what spent the 5/6 glide net.
pub const HIDDEN_MAX: usize = HIDDEN_START;
/// tanh deadband before fins move (same idea as throttle).
pub const FIN_DEADBAND: f64 = 0.08;
/// |tanh|=1 maps to this fraction of ±28° so the rail is not the default.
pub const FIN_CMD_SCALE: f64 = 0.92;
/// Outputs zeroed by plane-lock and frozen in CMA on pad/2 km: gimbal_z,
/// fin yaw/roll, RCS xyz.
pub const PLANE_LOCK_OUTPUTS: [usize; 6] = [2, 4, 5, 7, 8, 9];
/// Policy tick. Physics may run faster; the last action is held.
pub const POLICY_DT: f64 = 0.10;
/// asinh characteristic length — one scale for every curriculum stage.
pub const OBS_POS_M: f64 = 400.0;
pub const OBS_VEL_MPS: f64 = 80.0;
pub const OBS_RATE_RAD_S: f64 = 1.0;

/// Layout: W1 (H×N_IN) | b1 (H) | W2 (N_OUT×H) | b2 (N_OUT).
pub fn n_weights(hidden: usize) -> usize {
    hidden * N_IN + hidden + N_OUT * hidden + N_OUT
}

pub fn hidden_from_len(n: usize) -> usize {
    let den = N_IN + 1 + N_OUT;
    if n < n_weights(HIDDEN_START) {
        return HIDDEN_START;
    }
    let h = (n.saturating_sub(N_OUT)) / den;
    h.clamp(1, HIDDEN_MAX)
}

pub fn mlp_forward(w: &[f64], hidden: usize, x: &[f64; N_IN]) -> [f64; N_OUT] {
    let mut y = [0.0; N_OUT];
    let need = n_weights(hidden);
    if w.len() < need || hidden == 0 {
        return y;
    }
    let w1 = 0;
    let b1 = hidden * N_IN;
    let w2 = b1 + hidden;
    let b2 = w2 + N_OUT * hidden;
    let mut hact = vec![0.0; hidden];
    for j in 0..hidden {
        let mut s = w[b1 + j];
        let row = w1 + j * N_IN;
        for i in 0..N_IN {
            s += w[row + i] * x[i];
        }
        hact[j] = tanh(s);
    }
    for a in 0..N_OUT {
        let mut s = w[b2 + a];
        let row = w2 + a * hidden;
        for j in 0..hidden {
            s += w[row + j] * hact[j];
        }
        y[a] = tanh(s);
    }
    y
}

pub fn w2_start(hidden: usize) -> usize {
    hidden * N_IN + hidden
}

pub fn b2_start(hidden: usize) -> usize {
    w2_start(hidden) + N_OUT * hidden
}

/// True at indices that must stay 0 while pad/2 km plane-lock is on.
pub fn plane_lock_weight_mask(hidden: usize) -> Vec<bool> {
    let n = n_weights(hidden);
    let mut m = vec![false; n];
    let w2 = w2_start(hidden);
    let b2 = b2_start(hidden);
    for &a in &PLANE_LOCK_OUTPUTS {
        for j in 0..hidden {
            let i = w2 + a * hidden + j;
            if i < n {
                m[i] = true;
            }
        }
        if b2 + a < n {
            m[b2 + a] = true;
        }
    }
    m
}

pub fn apply_plane_lock_mask(w: &mut [f64], hidden: usize) {
    let mask = plane_lock_weight_mask(hidden);
    for (i, freeze) in mask.iter().enumerate() {
        if *freeze && i < w.len() {
            w[i] = 0.0;
        }
    }
}

fn fin_from_tanh(y: f64) -> f64 {
    if y.abs() <= FIN_DEADBAND {
        return 0.0;
    }
    let mag = (y.abs() - FIN_DEADBAND) / (1.0 - FIN_DEADBAND);
    y.signum() * mag * FIN_CMD_SCALE * FIN_MAX_DEFLECT_RAD
}

/// Pad-ENU engine position, velocity, body→ENU quaternion, body rate, and
/// fuel / 40 t. asinh keeps pad and RTLS in the same linear-ish band.
/// No q, Mach, AoA, phase, or predicted miss.
pub fn observe(nav: &Nav, body_y_ecef: Vec3, omega_body: Vec3) -> [f64; N_IN] {
    let bx = Vec3::new(
        nav.body_x_ecef.dot(nav.east),
        nav.body_x_ecef.dot(nav.north),
        nav.body_x_ecef.dot(nav.up),
    );
    let by = Vec3::new(
        body_y_ecef.dot(nav.east),
        body_y_ecef.dot(nav.north),
        body_y_ecef.dot(nav.up),
    );
    let mut bz = bx.cross(by);
    if bz.norm() < 1e-8 {
        bz = Vec3::Z;
    } else {
        bz = bz.normalized();
    }
    let by = bz.cross(bx);
    let by = if by.norm() < 1e-8 {
        Vec3::Y
    } else {
        by.normalized()
    };
    let bx = if bx.norm() < 1e-8 {
        Vec3::X
    } else {
        bx.normalized()
    };
    let q = Quat::from_axes(bx, by, bz).hemisphere();
    let engine = nav.pos_enu - bx * (STAGE_LENGTH_M * 0.5);
    [
        asinh(engine.x / OBS_POS_M),
        asinh(engine.y / OBS_POS_M),
        asinh(nav.engine_alt / OBS_POS_M),
        asinh(nav.v_enu.x / OBS_VEL_MPS),
        asinh(nav.v_enu.y / OBS_VEL_MPS),
        asinh(nav.v_enu.z / OBS_VEL_MPS),
        q.w,
        q.x,
        q.y,
        q.z,
        asinh(omega_body.x / OBS_RATE_RAD_S),
        asinh(omega_body.y / OBS_RATE_RAD_S),
        asinh(omega_body.z / OBS_RATE_RAD_S),
        (nav.fuel / START_FUEL_KG).clamp(0.0, 2.0),
    ]
}

pub fn actions_from_mlp(y: &[f64; N_OUT], plane_lock: bool) -> Controls {
    let mut u = Controls::default();
    if y[0] > 0.08 {
        u.throttle = saturate(THROTTLE_MIN + (y[0] - 0.08) / 0.92 * (THROTTLE_MAX - THROTTLE_MIN));
        u.n_engines = if y[6] > 0.20 {
            N_ENGINES_ENTRY
        } else {
            N_ENGINES_LANDING
        };
    }
    u.gimbal_y = clamp(y[1] * GIMBAL_MAX_RAD, -GIMBAL_MAX_RAD, GIMBAL_MAX_RAD);
    u.gimbal_z = clamp(y[2] * GIMBAL_MAX_RAD, -GIMBAL_MAX_RAD, GIMBAL_MAX_RAD);
    u.fin_pitch = clamp(fin_from_tanh(y[3]), -FIN_MAX_DEFLECT_RAD, FIN_MAX_DEFLECT_RAD);
    u.fin_yaw = clamp(fin_from_tanh(y[4]), -FIN_MAX_DEFLECT_RAD, FIN_MAX_DEFLECT_RAD);
    u.fin_roll = clamp(fin_from_tanh(y[5]), -FIN_MAX_DEFLECT_RAD, FIN_MAX_DEFLECT_RAD);
    u.rcs_x = clamp(y[7], -1.0, 1.0);
    u.rcs_y = clamp(y[8], -1.0, 1.0);
    u.rcs_z = clamp(y[9], -1.0, 1.0);
    if plane_lock {
        u.gimbal_z = 0.0;
        u.fin_yaw = 0.0;
        u.fin_roll = 0.0;
        u.rcs_x = 0.0;
        u.rcs_y = 0.0;
        u.rcs_z = 0.0;
    }
    u
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::earth::pad_ecef;
    use crate::guidance::nav_from;
    use crate::math::Vec3;
    use crate::scenario::Scenario;

    #[test]
    fn weight_count_matches_layout() {
        assert_eq!(n_weights(8), 8 * 14 + 8 + 10 * 8 + 10);
        assert_eq!(n_weights(8), 210);
        assert_eq!(hidden_from_len(210), 8);
        assert_eq!(n_weights(HIDDEN_MAX), 210);
    }

    #[test]
    fn zero_net_does_not_light_engines() {
        let w = vec![0.0; n_weights(8)];
        let x = [0.0; N_IN];
        let y = mlp_forward(&w, 8, &x);
        let u = actions_from_mlp(&y, false);
        assert_eq!(u.n_engines, 0);
        assert_eq!(u.throttle, 0.0);
        assert_eq!(u.gimbal_y, 0.0);
        assert_eq!(u.gimbal_z, 0.0);
        assert_eq!(u.rcs_x, 0.0);
        assert_eq!(u.rcs_y, 0.0);
        assert_eq!(u.rcs_z, 0.0);
        for v in y {
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn cluster_output_selects_three() {
        let mut y = [0.0; N_OUT];
        y[0] = 0.5;
        y[6] = 0.5;
        let u = actions_from_mlp(&y, false);
        assert_eq!(u.n_engines, 3);
        y[6] = 0.0;
        let u1 = actions_from_mlp(&y, false);
        assert_eq!(u1.n_engines, 1);
    }

    #[test]
    fn plane_lock_zeros_out_of_plane_actuators() {
        let y = [0.5, 0.4, 0.9, 0.3, 0.8, -0.7, 0.0, 0.9, 0.4, -0.8];
        let u = actions_from_mlp(&y, true);
        assert!(u.throttle > 0.3);
        assert_eq!(u.n_engines, 1);
        assert!(u.gimbal_y.abs() > 0.01);
        assert_eq!(u.gimbal_z, 0.0);
        assert_eq!(u.fin_yaw, 0.0);
        assert_eq!(u.fin_roll, 0.0);
        assert!(u.fin_pitch.abs() > 0.01);
        assert_eq!(u.rcs_x, 0.0);
        assert_eq!(u.rcs_y, 0.0);
        assert_eq!(u.rcs_z, 0.0);
    }

    #[test]
    fn observe_ignores_aero_scalars() {
        let pad = pad_ecef();
        let mut n0 = nav_from(
            pad + Vec3::new(0.0, 0.0, 100.0),
            Vec3::new(1.0, 0.0, -5.0),
            Vec3::new(0.0, 0.0, 1.0),
            100.0,
            80.0,
            1_000.0,
            0.1,
            0.3,
            1_000.0,
            wet_mass(1_000.0),
        );
        n0.east = Vec3::X;
        n0.north = Vec3::Y;
        n0.up = Vec3::Z;
        n0.body_x_ecef = Vec3::Z;
        let n1 = {
            let mut n = n0;
            n.q = 80_000.0;
            n.aoa = 0.9;
            n.mach = 4.0;
            n
        };
        let y = Vec3::Y;
        let w = Vec3::ZERO;
        let a = observe(&n0, y, w);
        let b = observe(&n1, y, w);
        for i in 0..N_IN {
            assert!((a[i] - b[i]).abs() < 1e-12, "feat {i}");
        }
    }

    #[test]
    fn asinh_obs_has_pad_slope_and_rtls_bound() {
        let pad = pad_ecef();
        let mut near = nav_from(
            pad + Vec3::new(0.0, 0.0, 80.0),
            Vec3::ZERO,
            Vec3::Z,
            80.0,
            80.0,
            0.0,
            0.0,
            0.0,
            1_000.0,
            wet_mass(1_000.0),
        );
        near.east = Vec3::X;
        near.north = Vec3::Y;
        near.up = Vec3::Z;
        near.body_x_ecef = Vec3::Z;
        near.pos_enu = Vec3::new(0.0, 0.0, 80.0);
        near.engine_alt = 80.0;
        let a = observe(&near, Vec3::Y, Vec3::ZERO);
        let mut far = near;
        far.engine_alt = 80_000.0;
        far.pos_enu.z = 80_000.0;
        let b = observe(&far, Vec3::Y, Vec3::ZERO);
        assert!(a[2].abs() > 0.05, "pad alt should have slope {}", a[2]);
        assert!(b[2].abs() < 8.0, "RTLS alt should stay bounded {}", b[2]);
        assert!(b[2].abs() > a[2].abs() + 1.0);
        let _ = Scenario::Rtls;
    }

    #[test]
    fn fin_deadband_holds_zero() {
        let mut y = [0.0; N_OUT];
        y[3] = 0.05;
        y[4] = -0.05;
        let u = actions_from_mlp(&y, false);
        assert_eq!(u.fin_pitch, 0.0);
        assert_eq!(u.fin_yaw, 0.0);
    }

    #[test]
    fn plane_lock_mask_covers_locked_outputs() {
        let m = plane_lock_weight_mask(8);
        assert_eq!(m.iter().filter(|b| **b).count(), 6 * (8 + 1));
        let mut w = vec![1.0; n_weights(8)];
        apply_plane_lock_mask(&mut w, 8);
        let y = mlp_forward(&w, 8, &[0.2; N_IN]);
        assert!(y[2].abs() < 1e-12);
        assert!(y[4].abs() < 1e-12);
        assert!(y[5].abs() < 1e-12);
        assert!(y[7].abs() < 1e-12);
    }
}
