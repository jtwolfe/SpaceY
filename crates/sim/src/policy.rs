//! CMA-NeuroES policy: a small tanh MLP that *is* the controller.
//!
//! Observations are pad-frame pose, velocity, body rate, and fuel. Actions
//! are throttle, TVC gimbal, and grid fins. Zero weights keep engines off
//! (deadband around tanh 0). No inner attitude PD, no RCS, no reference path.

use crate::constants::*;
use crate::guidance::{Controls, Nav};
use crate::math::{clamp, saturate, tanh, Quat, Vec3};
use crate::scenario::Scenario;

pub const N_IN: usize = 14;
pub const N_OUT: usize = 6;
pub const HIDDEN_START: usize = 8;
pub const HIDDEN_MAX: usize = 24;
/// Policy tick. Physics may run faster; the last action is held.
pub const POLICY_DT: f64 = 0.10;

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

/// New hidden unit starts at 0 so the old mapping is unchanged (CMA-TWEANN).
pub fn expand_hidden(w: &[f64], h_old: usize) -> Vec<f64> {
    let h_new = (h_old + 1).min(HIDDEN_MAX);
    if h_new == h_old {
        return w.to_vec();
    }
    let mut out = vec![0.0; n_weights(h_new)];
    for j in 0..h_old {
        let src = j * N_IN;
        let dst = j * N_IN;
        if src + N_IN <= w.len() {
            out[dst..dst + N_IN].copy_from_slice(&w[src..src + N_IN]);
        }
    }
    let b1_old = h_old * N_IN;
    let b1_new = h_new * N_IN;
    for j in 0..h_old {
        if b1_old + j < w.len() {
            out[b1_new + j] = w[b1_old + j];
        }
    }
    let w2_old = b1_old + h_old;
    let w2_new = b1_new + h_new;
    for a in 0..N_OUT {
        let src = w2_old + a * h_old;
        let dst = w2_new + a * h_new;
        for j in 0..h_old {
            if src + j < w.len() {
                out[dst + j] = w[src + j];
            }
        }
    }
    let b2_old = w2_old + N_OUT * h_old;
    let b2_new = w2_new + N_OUT * h_new;
    for a in 0..N_OUT {
        if b2_old + a < w.len() {
            out[b2_new + a] = w[b2_old + a];
        }
    }
    out
}

fn obs_scales(scenario: Scenario) -> (f64, f64, f64) {
    match scenario {
        Scenario::Pad => (120.0, 40.0, 1.0),
        Scenario::Slam | Scenario::Attitude | Scenario::Wind => (500.0, 80.0, 1.2),
        Scenario::Glide => (8_000.0, 200.0, 1.5),
        Scenario::Rtls => (12_000.0, 400.0, 1.5),
    }
}

/// Pad-ENU engine position, velocity, body→ENU quaternion, body rate, fuel.
/// No q, Mach, AoA, phase, or predicted miss.
pub fn observe(
    nav: &Nav,
    body_y_ecef: Vec3,
    omega_body: Vec3,
    scenario: Scenario,
) -> [f64; N_IN] {
    let (pos_s, vel_s, rate_s) = obs_scales(scenario);
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
    let fuel_s = scenario.start_fuel().max(1.0);
    [
        engine.x / pos_s,
        engine.y / pos_s,
        nav.engine_alt / pos_s,
        nav.v_enu.x / vel_s,
        nav.v_enu.y / vel_s,
        nav.v_enu.z / vel_s,
        q.w,
        q.x,
        q.y,
        q.z,
        omega_body.x / rate_s,
        omega_body.y / rate_s,
        omega_body.z / rate_s,
        nav.fuel / fuel_s,
    ]
}

pub fn actions_from_mlp(y: &[f64; N_OUT], plane_lock: bool) -> Controls {
    let mut u = Controls::default();
    if y[0] > 0.08 {
        u.throttle = saturate(THROTTLE_MIN + (y[0] - 0.08) / 0.92 * (THROTTLE_MAX - THROTTLE_MIN));
        u.n_engines = N_ENGINES_LANDING;
    }
    u.gimbal_y = clamp(y[1] * GIMBAL_MAX_RAD, -GIMBAL_MAX_RAD, GIMBAL_MAX_RAD);
    u.gimbal_z = clamp(y[2] * GIMBAL_MAX_RAD, -GIMBAL_MAX_RAD, GIMBAL_MAX_RAD);
    u.fin_pitch = clamp(y[3] * FIN_MAX_DEFLECT_RAD, -FIN_MAX_DEFLECT_RAD, FIN_MAX_DEFLECT_RAD);
    u.fin_yaw = clamp(y[4] * FIN_MAX_DEFLECT_RAD, -FIN_MAX_DEFLECT_RAD, FIN_MAX_DEFLECT_RAD);
    u.fin_roll = clamp(y[5] * FIN_MAX_DEFLECT_RAD, -FIN_MAX_DEFLECT_RAD, FIN_MAX_DEFLECT_RAD);
    if plane_lock {
        u.gimbal_z = 0.0;
        u.fin_yaw = 0.0;
        u.fin_roll = 0.0;
    }
    u
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::earth::pad_ecef;
    use crate::guidance::nav_from;
    use crate::math::Vec3;

    #[test]
    fn weight_count_matches_layout() {
        assert_eq!(n_weights(8), 8 * 14 + 8 + 6 * 8 + 6);
        assert_eq!(n_weights(8), 174);
        assert_eq!(hidden_from_len(174), 8);
        assert_eq!(hidden_from_len(n_weights(16)), 16);
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
        for v in y {
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn plane_lock_zeros_out_of_plane_actuators() {
        let y = [0.5, 0.4, 0.9, 0.3, 0.8, -0.7];
        let u = actions_from_mlp(&y, true);
        assert!(u.throttle > 0.3);
        assert_eq!(u.n_engines, 1);
        assert!(u.gimbal_y.abs() > 0.01);
        assert_eq!(u.gimbal_z, 0.0);
        assert_eq!(u.fin_yaw, 0.0);
        assert_eq!(u.fin_roll, 0.0);
        assert!(u.fin_pitch.abs() > 0.01);
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
        let a = observe(&n0, y, w, Scenario::Pad);
        let b = observe(&n1, y, w, Scenario::Pad);
        for i in 0..N_IN {
            assert!((a[i] - b[i]).abs() < 1e-12, "feat {i}");
        }
    }

    #[test]
    fn expand_preserves_forward() {
        let mut w = vec![0.0; n_weights(8)];
        w[0] = 0.4;
        w[n_weights(8) - 1] = -0.2;
        let x = [
            0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.5,
        ];
        let y0 = mlp_forward(&w, 8, &x);
        let w2 = expand_hidden(&w, 8);
        assert_eq!(w2.len(), n_weights(9));
        let y1 = mlp_forward(&w2, 9, &x);
        for i in 0..N_OUT {
            assert!(
                (y0[i] - y1[i]).abs() < 1e-12,
                "out {i}: {} vs {}",
                y0[i],
                y1[i]
            );
        }
    }
}
