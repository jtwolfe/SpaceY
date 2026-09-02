//! Rotating Earth: WGS-84 geodesy, ECI ↔ ECEF, J2 gravity.

use crate::constants::{
    EARTH_E2, EARTH_J2, EARTH_MU, EARTH_OMEGA, EARTH_RADIUS_EQ, PAD_ALT_M, PAD_LAT_DEG, PAD_LON_DEG,
};
use crate::math::{atan, atan2, cos, sin, sqrt, Quat, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct Geodetic {
    pub lat: f64,
    pub lon: f64,
    pub alt: f64,
}

pub fn deg(d: f64) -> f64 {
    d * std::f64::consts::PI / 180.0
}

/// ECI → ECEF rotation about +Z. Earth spins *east*, so the ECEF axes
/// rotate west relative to inertial: angle = −ω t. At t = 0 the frames
/// coincide.
///
/// The opposite sign made `d(r_ecef)/dt = v_ground + 2 ω × r`. A stack
/// that had killed pad-ENU velocity still walked ~0.8 km/s east — the
/// #4 50 km dry slide after a pad-theater intercept.
pub fn earth_angle(t: f64) -> f64 {
    -EARTH_OMEGA * t
}

/// ECI and ECEF share +Z (Earth axis). At t = 0 the frames coincide.
pub fn q_eci_to_ecef(t: f64) -> Quat {
    Quat::from_axis_angle(Vec3::Z, earth_angle(t))
}

pub fn eci_to_ecef(r_eci: Vec3, t: f64) -> Vec3 {
    q_eci_to_ecef(t).rotate(r_eci)
}

pub fn ecef_to_eci(r_ecef: Vec3, t: f64) -> Vec3 {
    q_eci_to_ecef(t).conjugate().rotate(r_ecef)
}

/// Inertial velocity expressed in ECEF axes, minus Earth spin → ground-relative.
pub fn eci_vel_to_ecef_ground(r_eci: Vec3, v_eci: Vec3, t: f64) -> (Vec3, Vec3) {
    let r_ecef = eci_to_ecef(r_eci, t);
    let v_inertial_ecef = q_eci_to_ecef(t).rotate(v_eci);
    let omega = Vec3::new(0.0, 0.0, EARTH_OMEGA);
    let v_ground = v_inertial_ecef - omega.cross(r_ecef);
    (r_ecef, v_ground)
}

pub fn geodetic_to_ecef(g: Geodetic) -> Vec3 {
    let sin_lat = sin(g.lat);
    let cos_lat = cos(g.lat);
    let n = EARTH_RADIUS_EQ / sqrt(1.0 - EARTH_E2 * sin_lat * sin_lat);
    Vec3::new(
        (n + g.alt) * cos_lat * cos(g.lon),
        (n + g.alt) * cos_lat * sin(g.lon),
        (n * (1.0 - EARTH_E2) + g.alt) * sin_lat,
    )
}

pub fn ecef_to_geodetic(r: Vec3) -> Geodetic {
    let lon = atan2(r.y, r.x);
    let p = sqrt(r.x * r.x + r.y * r.y);
    let mut lat = atan(r.z / p.max(1e-9));
    for _ in 0..8 {
        let s = sin(lat);
        let n = EARTH_RADIUS_EQ / sqrt(1.0 - EARTH_E2 * s * s);
        lat = atan2(r.z + EARTH_E2 * n * s, p);
    }
    let s = sin(lat);
    let n = EARTH_RADIUS_EQ / sqrt(1.0 - EARTH_E2 * s * s);
    let alt = if cos(lat).abs() > 1e-8 {
        p / cos(lat) - n
    } else {
        r.z.abs() - n * (1.0 - EARTH_E2)
    };
    Geodetic { lat, lon, alt }
}

/// Local ENU basis in ECEF components at `lat`, `lon`.
pub fn enu_basis(lat: f64, lon: f64) -> (Vec3, Vec3, Vec3) {
    let sin_lat = sin(lat);
    let cos_lat = cos(lat);
    let sin_lon = sin(lon);
    let cos_lon = cos(lon);
    let east = Vec3::new(-sin_lon, cos_lon, 0.0);
    let north = Vec3::new(-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat);
    let up = Vec3::new(cos_lat * cos_lon, cos_lat * sin_lon, sin_lat);
    (east, north, up)
}

pub fn ecef_to_enu(r_ecef: Vec3, origin_ecef: Vec3, lat: f64, lon: f64) -> Vec3 {
    let d = r_ecef - origin_ecef;
    let (e, n, u) = enu_basis(lat, lon);
    Vec3::new(d.dot(e), d.dot(n), d.dot(u))
}

pub fn enu_to_ecef_vec(v: Vec3, lat: f64, lon: f64) -> Vec3 {
    let (e, n, u) = enu_basis(lat, lon);
    e * v.x + n * v.y + u * v.z
}

/// Great-circle distance between two ECEF (or ECI-at-t0) position vectors.
pub fn great_circle_m(a: Vec3, b: Vec3) -> f64 {
    EARTH_RADIUS_EQ * crate::math::angle_between(a, b)
}

/// Keplerian periapsis radius from an ECI (or inertial) state.
/// Hyperbolic / near-parabolic trajectories return `f64::INFINITY`.
pub fn periapsis_radius(r: Vec3, v: Vec3) -> f64 {
    let r1 = r.norm();
    if r1 < 1.0 {
        return 0.0;
    }
    let mu = EARTH_MU;
    let eps = 0.5 * v.norm_squared() - mu / r1;
    let h = r.cross(v).norm();
    if eps >= 0.0 {
        return f64::INFINITY;
    }
    let a = -mu / (2.0 * eps);
    if !a.is_finite() || a <= 0.0 {
        return f64::INFINITY;
    }
    let arg = 1.0 + 2.0 * eps * h * h / (mu * mu);
    let e = sqrt(arg.max(0.0));
    a * (1.0 - e)
}

/// J2 gravity in a frame whose +Z is Earth's rotation axis (ECI or ECEF).
pub fn gravity_j2(r: Vec3) -> Vec3 {
    let r2 = r.norm_squared();
    let r1 = sqrt(r2);
    if r1 < 1e5 {
        return Vec3::ZERO;
    }
    let z2r2 = r.z * r.z / r2;
    let f = 1.5 * EARTH_J2 * (EARTH_RADIUS_EQ / r1).powi(2);
    let mu_r3 = EARTH_MU / (r2 * r1);
    Vec3::new(
        -mu_r3 * r.x * (1.0 + f * (1.0 - 5.0 * z2r2)),
        -mu_r3 * r.y * (1.0 + f * (1.0 - 5.0 * z2r2)),
        -mu_r3 * r.z * (1.0 + f * (3.0 - 5.0 * z2r2)),
    )
}

pub fn pad_geodetic() -> Geodetic {
    Geodetic {
        lat: deg(PAD_LAT_DEG),
        lon: deg(PAD_LON_DEG),
        alt: PAD_ALT_M,
    }
}

pub fn pad_ecef() -> Vec3 {
    geodetic_to_ecef(pad_geodetic())
}

/// Tilt of body +X away from local up (0 = engines down, interstage skyward).
pub fn tilt_from_vertical(body_x_ecef: Vec3, up: Vec3) -> f64 {
    crate::math::angle_between(body_x_ecef, up)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{EARTH_MU, EARTH_RADIUS_EQ};

    #[test]
    fn geodetic_roundtrip_pad() {
        let g = pad_geodetic();
        let r = geodetic_to_ecef(g);
        let back = ecef_to_geodetic(r);
        assert!((back.lat - g.lat).abs() < 1e-8);
        assert!((back.lon - g.lon).abs() < 1e-8);
        assert!((back.alt - g.alt).abs() < 0.5);
    }

    #[test]
    fn frames_coincide_at_t0() {
        let r = Vec3::new(1.0, 2.0, 3.0);
        let r2 = eci_to_ecef(r, 0.0);
        assert!((r2 - r).norm() < 1e-12);
    }

    #[test]
    fn gravity_points_inward() {
        let r = geodetic_to_ecef(pad_geodetic());
        let a = gravity_j2(r);
        assert!(a.dot(r) < 0.0);
        assert!((a.norm() - 9.8).abs() < 0.15);
    }

    #[test]
    fn circular_periapsis_is_radius() {
        let r = Vec3::new(EARTH_RADIUS_EQ + 220_000.0, 0.0, 0.0);
        let v_c = crate::math::sqrt(EARTH_MU / r.norm());
        let v = Vec3::new(0.0, v_c, 0.0);
        let rp = periapsis_radius(r, v);
        assert!((rp - r.norm()).abs() < 2_000.0, "rp={rp} r={}", r.norm());
    }

    #[test]
    fn ground_fixed_pad_stays_in_ecef() {
        let pad = pad_ecef();
        let omega = Vec3::new(0.0, 0.0, EARTH_OMEGA);
        for t in [0.0, 10.0, 200.0, 2_520.0] {
            let r_eci = ecef_to_eci(pad, t);
            // Inertial velocity of a point fixed on the rotating Earth.
            let v_eci = omega.cross(r_eci);
            let (r_ecef, v_g) = eci_vel_to_ecef_ground(r_eci, v_eci, t);
            assert!(
                (r_ecef - pad).norm() < 1.0,
                "t={t} drift={} m",
                (r_ecef - pad).norm()
            );
            assert!(v_g.norm() < 0.05, "t={t} v_ground={} m/s", v_g.norm());
        }
    }

    #[test]
    fn ecef_position_tracks_ground_velocity() {
        // Finite-difference: a free ballistic step must move r_ecef at v_ground,
        // not at v_ground + 2 ω × r (~0.8 km/s east at the Cape).
        let pad = pad_ecef();
        let (east, _, up) = enu_basis(pad_geodetic().lat, pad_geodetic().lon);
        let r_ecef0 = pad + up * 1_200.0 + east * 400.0;
        let v_ground0 = east * 30.0 + up * -80.0;
        let t = 2_780.0;
        let r_eci = ecef_to_eci(r_ecef0, t);
        let v_inertial_ecef = v_ground0 + Vec3::new(0.0, 0.0, EARTH_OMEGA).cross(r_ecef0);
        let v_eci = q_eci_to_ecef(t).conjugate().rotate(v_inertial_ecef);
        let dt = 0.20;
        let (_, vg) = eci_vel_to_ecef_ground(r_eci, v_eci, t);
        let r_eci1 = r_eci + v_eci * dt;
        let (r1, _) = eci_vel_to_ecef_ground(r_eci1, v_eci, t + dt);
        let dr = r1 - r_ecef0;
        let err = (dr - vg * dt).norm();
        assert!(
            err < 8.0,
            "pos/vel inconsistency {err} m (dr={}, vg dt={})",
            dr.norm(),
            vg.norm() * dt
        );
        let enu0 = ecef_to_enu(r_ecef0, pad, pad_geodetic().lat, pad_geodetic().lon);
        let enu1 = ecef_to_enu(r1, pad, pad_geodetic().lat, pad_geodetic().lon);
        assert!(
            (enu1.x - enu0.x - 30.0 * dt).abs() < 4.0,
            "east walk {} vs 30 m/s",
            (enu1.x - enu0.x) / dt
        );
    }
}
