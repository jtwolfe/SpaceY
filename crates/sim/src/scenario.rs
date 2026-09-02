//! Selectable recovery scenarios. RTLS-class first-stage reentry is the default;
//! LEO deorbit is the same F9-class vehicle started at orbital energy.

use crate::constants::*;
use crate::earth::{
    ecef_to_geodetic, enu_basis, geodetic_to_ecef, pad_geodetic, Geodetic,
};
use crate::math::{Quat, Vec3};
use rand::Rng;

/// `0` = RTLS first-stage reentry (default). `1` = LEO-class deorbit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Scenario {
    #[default]
    Rtls,
    LeoDeorbit,
}

impl Scenario {
    pub fn from_id(id: u32) -> Self {
        match id {
            1 => Scenario::LeoDeorbit,
            _ => Scenario::Rtls,
        }
    }

    pub fn id(self) -> u32 {
        match self {
            Scenario::Rtls => 0,
            Scenario::LeoDeorbit => 1,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Scenario::Rtls => "rtls",
            Scenario::LeoDeorbit => "leo",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Scenario::Rtls => "RTLS reentry",
            Scenario::LeoDeorbit => "LEO deorbit",
        }
    }

    pub fn timeout(self) -> f64 {
        match self {
            Scenario::Rtls => RTLS_TIMEOUT_S,
            Scenario::LeoDeorbit => LEO_TIMEOUT_S,
        }
    }

    pub fn start_fuel(self) -> f64 {
        match self {
            Scenario::Rtls => START_FUEL_KG,
            Scenario::LeoDeorbit => ORBITAL_START_FUEL_KG,
        }
    }

    pub fn is_orbital(self) -> bool {
        matches!(self, Scenario::LeoDeorbit)
    }

    pub fn spawn(self, rng: &mut impl Rng) -> Spawn {
        match self {
            Scenario::Rtls => spawn_rtls(rng),
            Scenario::LeoDeorbit => spawn_leo(rng),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Spawn {
    pub r_eci: Vec3,
    pub v_eci: Vec3,
    pub q_body_to_eci: Quat,
    pub fuel: f64,
    pub start_ecef: Vec3,
}

/// Current v1 start: ~80 km, ~2.05 km/s Earth-relative, westbound toward LZ-1.
fn spawn_rtls(rng: &mut impl Rng) -> Spawn {
    let pad = pad_geodetic();
    let dlon = 62_000.0 / (EARTH_RADIUS_EQ * pad.lat.cos());
    let jitter = 0.0008 * (rng.gen::<f64>() - 0.5);
    let geo = Geodetic {
        lat: pad.lat + 0.0004 * (rng.gen::<f64>() - 0.5),
        lon: pad.lon + dlon + jitter,
        alt: 80_000.0 + 1_500.0 * (rng.gen::<f64>() - 0.5),
    };
    let r_ecef = geodetic_to_ecef(geo);
    let (east, north, up) = enu_basis(geo.lat, geo.lon);
    let speed = 2_050.0 + 60.0 * (rng.gen::<f64>() - 0.5);
    let gamma = (-7.5 + rng.gen::<f64>() * 0.8) * std::f64::consts::PI / 180.0;
    let heading = (269.2 + rng.gen::<f64>() * 1.2) * std::f64::consts::PI / 180.0;
    let vh = speed * gamma.cos();
    let vu = speed * gamma.sin();
    let v_enu = Vec3::new(vh * heading.sin(), vh * heading.cos(), vu);
    let v_ground_ecef = east * v_enu.x + north * v_enu.y + up * v_enu.z;
    let omega_e = Vec3::new(0.0, 0.0, EARTH_OMEGA);
    let v_eci = v_ground_ecef + omega_e.cross(r_ecef);
    let desired_x = if v_ground_ecef.norm() > 1.0 {
        -v_ground_ecef.normalized()
    } else {
        up
    };
    let q = Quat::from_rotation_arc(Vec3::X, desired_x);
    Spawn {
        r_eci: r_ecef,
        v_eci,
        q_body_to_eci: q,
        fuel: START_FUEL_KG,
        start_ecef: r_ecef,
    }
}

/// Circular LEO-class state (~220 km, ~7.8 km/s inertial) on a plane that
/// overflies LZ-1 after a retrograde deorbit + half-rev coast.
fn spawn_leo(rng: &mut impl Rng) -> Spawn {
    let pad = pad_geodetic();
    // Half-period of a 220 km circular orbit is ~44 min; Earth rotates ~11°.
    let coast = 2_520.0 + 30.0 * (rng.gen::<f64>() - 0.5);
    let lead = EARTH_OMEGA * coast;
    let peri_geo = Geodetic {
        lat: pad.lat + 0.002 * (rng.gen::<f64>() - 0.5),
        lon: pad.lon - lead + 0.004 * (rng.gen::<f64>() - 0.5),
        alt: DEORBIT_PERI_TARGET_M,
    };
    let r_peri = geodetic_to_ecef(peri_geo);
    let (east, _, _) = enu_basis(peri_geo.lat, peri_geo.lon);
    let h_hat = r_peri.cross(east).normalized();

    // Just after apogee of the *post-deorbit* ellipse: ~175° of true anomaly
    // before periapsis. The start itself is circular at 220 km.
    let nu = (174.0 + rng.gen::<f64>() * 3.0) * std::f64::consts::PI / 180.0;
    let r_dir = Quat::from_axis_angle(h_hat, -nu).rotate(r_peri.normalized());
    let alt = ORBITAL_ALT_M + 2_500.0 * (rng.gen::<f64>() - 0.5);
    let probe = r_dir * (EARTH_RADIUS_EQ + alt);
    let g = ecef_to_geodetic(probe);
    let r_ecef = geodetic_to_ecef(Geodetic {
        lat: g.lat,
        lon: g.lon,
        alt,
    });

    let r1 = r_ecef.norm();
    let v_circ = (EARTH_MU / r1).sqrt();
    let v_dir = h_hat.cross(r_ecef.normalized());
    let v_eci = v_dir * v_circ;

    let desired_x = if v_eci.norm() > 1.0 {
        -v_eci.normalized()
    } else {
        r_ecef.normalized()
    };
    let q = Quat::from_rotation_arc(Vec3::X, desired_x);
    Spawn {
        r_eci: r_ecef,
        v_eci,
        q_body_to_eci: q,
        fuel: ORBITAL_START_FUEL_KG,
        start_ecef: r_ecef,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atmosphere;
    use crate::earth::{eci_vel_to_ecef_ground, periapsis_radius};
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn default_is_rtls() {
        assert_eq!(Scenario::default(), Scenario::Rtls);
        assert_eq!(Scenario::from_id(0), Scenario::Rtls);
        assert_eq!(Scenario::from_id(1), Scenario::LeoDeorbit);
    }

    #[test]
    fn leo_start_is_orbital_energy_in_vacuum() {
        let mut rng = SmallRng::seed_from_u64(7);
        let s = Scenario::LeoDeorbit.spawn(&mut rng);
        let geo = ecef_to_geodetic(s.r_eci);
        let air = atmosphere::lookup(geo.alt);
        let v_inertial = s.v_eci.norm();
        let (_, v_g) = eci_vel_to_ecef_ground(s.r_eci, s.v_eci, 0.0);
        assert!(
            geo.alt > 200_000.0,
            "LEO start alt {} m",
            geo.alt
        );
        assert!(
            v_inertial > 7_600.0 && v_inertial < 8_000.0,
            "inertial speed {} m/s",
            v_inertial
        );
        assert!(
            v_g.norm() > 7_100.0,
            "Earth-relative {} m/s",
            v_g.norm()
        );
        assert!(
            air.density < 1e-9,
            "near-vacuum density {}",
            air.density
        );
        let rp = periapsis_radius(s.r_eci, s.v_eci);
        assert!(
            rp > EARTH_RADIUS_EQ + 180_000.0,
            "circular-ish periapsis {rp}"
        );
    }

    #[test]
    fn rtls_start_is_suborbital() {
        let mut rng = SmallRng::seed_from_u64(1);
        let s = Scenario::Rtls.spawn(&mut rng);
        let geo = ecef_to_geodetic(s.r_eci);
        let (_, v_g) = eci_vel_to_ecef_ground(s.r_eci, s.v_eci, 0.0);
        assert!(geo.alt > 70_000.0 && geo.alt < 90_000.0);
        assert!(v_g.norm() > 1_800.0 && v_g.norm() < 2_400.0);
    }

    #[test]
    fn leo_start_is_half_rev_from_pad() {
        let mut rng = SmallRng::seed_from_u64(7);
        let s = Scenario::LeoDeorbit.spawn(&mut rng);
        let pad = crate::earth::pad_ecef();
        let ang = crate::math::angle_between(s.r_eci, pad);
        let gc = EARTH_RADIUS_EQ * ang;
        assert!(
            gc > 12_000_000.0,
            "great-circle to pad should be half-rev, got {gc} m"
        );
    }
}
