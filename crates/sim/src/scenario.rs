//! RTLS training stages. One mission — recover the booster to LZ-1.
//!
//! The net learns in slices (pad slam → 2 km → 6DOF → wind → glide → full
//! RTLS). The product is always RTLS; stages are not separate games.

use crate::constants::*;
use crate::earth::{enu_basis, geodetic_to_ecef, pad_geodetic, Geodetic};
use crate::math::{cos, sin, Quat, Vec3};
use rand::Rng;

pub const STAGE_COUNT: u32 = 6;

/// Curriculum slice of the RTLS landing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Scenario {
    #[default]
    Pad,
    Slam,
    Attitude,
    Wind,
    Glide,
    Rtls,
}

impl Scenario {
    pub fn from_id(id: u32) -> Self {
        match id {
            1 => Scenario::Slam,
            2 => Scenario::Attitude,
            3 => Scenario::Wind,
            4 => Scenario::Glide,
            5 => Scenario::Rtls,
            _ => Scenario::Pad,
        }
    }

    pub fn id(self) -> u32 {
        match self {
            Scenario::Pad => 0,
            Scenario::Slam => 1,
            Scenario::Attitude => 2,
            Scenario::Wind => 3,
            Scenario::Glide => 4,
            Scenario::Rtls => 5,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Scenario::Pad => "pad",
            Scenario::Slam => "slam",
            Scenario::Attitude => "attitude",
            Scenario::Wind => "wind",
            Scenario::Glide => "glide",
            Scenario::Rtls => "rtls",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Scenario::Pad => "pad slam",
            Scenario::Slam => "2 km slam",
            Scenario::Attitude => "6DOF",
            Scenario::Wind => "wind",
            Scenario::Glide => "glide",
            Scenario::Rtls => "RTLS",
        }
    }

    pub fn timeout(self) -> f64 {
        match self {
            Scenario::Pad => HOVER_TIMEOUT_S,
            Scenario::Slam | Scenario::Attitude | Scenario::Wind => SUICIDE_TIMEOUT_S,
            Scenario::Glide => GLIDE_TIMEOUT_S,
            Scenario::Rtls => RTLS_TIMEOUT_S,
        }
    }

    pub fn start_fuel(self) -> f64 {
        match self {
            Scenario::Pad => HOVER_FUEL_KG,
            Scenario::Slam | Scenario::Attitude | Scenario::Wind => SUICIDE_FUEL_KG,
            Scenario::Glide => GLIDE_FUEL_KG,
            Scenario::Rtls => START_FUEL_KG,
        }
    }

    /// Landing phase from t=0; no corridor.
    pub fn is_terminal_hop(self) -> bool {
        matches!(
            self,
            Scenario::Pad | Scenario::Slam | Scenario::Attitude | Scenario::Wind
        )
    }

    pub fn plane_lock(self) -> bool {
        matches!(self, Scenario::Pad | Scenario::Slam)
    }

    pub fn domain_rand_wind(self) -> bool {
        matches!(self, Scenario::Wind)
    }

    pub fn next_gate(self) -> Option<Scenario> {
        match self {
            Scenario::Pad => Some(Scenario::Slam),
            Scenario::Slam => Some(Scenario::Attitude),
            Scenario::Attitude => Some(Scenario::Wind),
            Scenario::Wind => Some(Scenario::Glide),
            Scenario::Glide => Some(Scenario::Rtls),
            Scenario::Rtls => None,
        }
    }

    pub fn spawn(self, rng: &mut impl Rng) -> Spawn {
        match self {
            Scenario::Pad => spawn_pad(rng),
            Scenario::Slam => spawn_slam_2d(rng),
            Scenario::Attitude | Scenario::Wind => spawn_slam_6dof(rng),
            Scenario::Glide => spawn_glide(rng),
            Scenario::Rtls => spawn_rtls(rng),
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

fn quat_body_to_world(body_x: Vec3, body_y_hint: Vec3) -> Quat {
    let x = body_x.normalized();
    let mut z = x.cross(body_y_hint);
    if z.norm() < 1e-8 {
        z = x.cross(Vec3::Y);
        if z.norm() < 1e-8 {
            z = x.cross(Vec3::Z);
        }
    }
    let z = z.normalized();
    let y = z.cross(x).normalized();
    Quat::from_axes(x, y, z)
}

fn spawn_upright(
    rng: &mut impl Rng,
    engine_alt: f64,
    v_enu: Vec3,
    east_m: f64,
    north_m: f64,
    fuel: f64,
    plane_lock: bool,
) -> Spawn {
    let pad = pad_geodetic();
    let (east, north, up) = enu_basis(pad.lat, pad.lon);
    let jitter_e = if plane_lock {
        east_m
    } else {
        east_m + 4.0 * (rng.gen::<f64>() - 0.5)
    };
    let jitter_n = if plane_lock {
        0.0
    } else {
        north_m + 4.0 * (rng.gen::<f64>() - 0.5)
    };
    let dlat = jitter_n / EARTH_RADIUS_EQ;
    let dlon = jitter_e / (EARTH_RADIUS_EQ * cos(pad.lat).max(0.3));
    let geo_alt = engine_alt + STAGE_LENGTH_M * 0.5 + 2.0 * (rng.gen::<f64>() - 0.5);
    let geo = Geodetic {
        lat: pad.lat + dlat,
        lon: pad.lon + dlon,
        alt: geo_alt,
    };
    let r_ecef = geodetic_to_ecef(geo);
    let v_use = if plane_lock {
        Vec3::new(v_enu.x, 0.0, v_enu.z)
    } else {
        v_enu
    };
    let v_ground_ecef = east * v_use.x + north * v_use.y + up * v_use.z;
    let omega_e = Vec3::new(0.0, 0.0, EARTH_OMEGA);
    let v_eci = v_ground_ecef + omega_e.cross(r_ecef);
    let q = quat_body_to_world(up, north);
    Spawn {
        r_eci: r_ecef,
        v_eci,
        q_body_to_eci: q,
        fuel,
        start_ecef: r_ecef,
    }
}

fn spawn_pad(rng: &mut impl Rng) -> Spawn {
    let alt = 82.0 + 8.0 * (rng.gen::<f64>() - 0.5);
    let vx = 1.5 * (rng.gen::<f64>() - 0.5);
    let vz = -1.5 * rng.gen::<f64>();
    let e = 6.0 * (rng.gen::<f64>() - 0.5);
    spawn_upright(rng, alt, Vec3::new(vx, 0.0, vz), e, 0.0, HOVER_FUEL_KG, true)
}

fn spawn_slam_2d(rng: &mut impl Rng) -> Spawn {
    let alt = 2_050.0 + 80.0 * (rng.gen::<f64>() - 0.5);
    let vx = 8.0 * (rng.gen::<f64>() - 0.5);
    let vz = -(40.0 + 80.0 * rng.gen::<f64>());
    let e = 200.0 * (rng.gen::<f64>() - 0.5);
    spawn_upright(
        rng,
        alt,
        Vec3::new(vx, 0.0, vz),
        e,
        0.0,
        SUICIDE_FUEL_KG,
        true,
    )
}

fn spawn_slam_6dof(rng: &mut impl Rng) -> Spawn {
    let alt = 2_050.0 + 80.0 * (rng.gen::<f64>() - 0.5);
    let vx = 8.0 * (rng.gen::<f64>() - 0.5);
    let vy = 8.0 * (rng.gen::<f64>() - 0.5);
    let vz = -(40.0 + 80.0 * rng.gen::<f64>());
    let e = 200.0 * (rng.gen::<f64>() - 0.5);
    let n = 200.0 * (rng.gen::<f64>() - 0.5);
    spawn_upright(
        rng,
        alt,
        Vec3::new(vx, vy, vz),
        e,
        n,
        SUICIDE_FUEL_KG,
        false,
    )
}

fn spawn_glide(rng: &mut impl Rng) -> Spawn {
    let pad = pad_geodetic();
    let dlon = 11_000.0 / (EARTH_RADIUS_EQ * cos(pad.lat).max(0.3));
    let geo = Geodetic {
        lat: pad.lat + 0.0003 * (rng.gen::<f64>() - 0.5),
        lon: pad.lon + dlon + 0.0004 * (rng.gen::<f64>() - 0.5),
        alt: 20_500.0 + 400.0 * (rng.gen::<f64>() - 0.5),
    };
    let r_ecef = geodetic_to_ecef(geo);
    let (east, north, up) = enu_basis(geo.lat, geo.lon);
    let speed = 380.0 + 20.0 * (rng.gen::<f64>() - 0.5);
    let gamma = (-9.0 + rng.gen::<f64>() * 1.2) * std::f64::consts::PI / 180.0;
    let heading = (268.5 + rng.gen::<f64>() * 2.0) * std::f64::consts::PI / 180.0;
    let vh = speed * cos(gamma);
    let vu = speed * sin(gamma);
    let v_enu = Vec3::new(vh * sin(heading), vh * cos(heading), vu);
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
        fuel: GLIDE_FUEL_KG,
        start_ecef: r_ecef,
    }
}

/// ~80 km, ~2.05 km/s Earth-relative, westbound toward LZ-1.
fn spawn_rtls(rng: &mut impl Rng) -> Spawn {
    let pad = pad_geodetic();
    let dlon = 62_000.0 / (EARTH_RADIUS_EQ * cos(pad.lat));
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
    let vh = speed * cos(gamma);
    let vu = speed * sin(gamma);
    let v_enu = Vec3::new(vh * sin(heading), vh * cos(heading), vu);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::earth::eci_vel_to_ecef_ground;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn stages_are_rtls_curriculum() {
        assert_eq!(Scenario::default(), Scenario::Pad);
        assert_eq!(Scenario::from_id(0), Scenario::Pad);
        assert_eq!(Scenario::from_id(5), Scenario::Rtls);
        assert!(Scenario::Pad.plane_lock());
        assert!(!Scenario::Attitude.plane_lock());
        assert!(Scenario::Wind.domain_rand_wind());
        assert_eq!(Scenario::Pad.next_gate(), Some(Scenario::Slam));
        assert_eq!(Scenario::Rtls.next_gate(), None);
        assert_eq!(STAGE_COUNT, 6);
    }

    #[test]
    fn rtls_start_is_suborbital() {
        let mut rng = SmallRng::seed_from_u64(1);
        let s = Scenario::Rtls.spawn(&mut rng);
        let geo = crate::earth::ecef_to_geodetic(s.r_eci);
        let (_, v_g) = eci_vel_to_ecef_ground(s.r_eci, s.v_eci, 0.0);
        assert!(geo.alt > 70_000.0 && geo.alt < 90_000.0);
        assert!(v_g.norm() > 1_800.0 && v_g.norm() < 2_400.0);
    }

    #[test]
    fn pad_starts_over_the_lz() {
        let mut rng = SmallRng::seed_from_u64(1);
        let s = Scenario::Pad.spawn(&mut rng);
        let geo = crate::earth::ecef_to_geodetic(s.r_eci);
        let pad = crate::earth::pad_ecef();
        let range = crate::earth::great_circle_m(s.r_eci, pad);
        assert!(geo.alt > 90.0 && geo.alt < 140.0, "alt {}", geo.alt);
        assert!(range < 80.0, "range {range}");
        assert!((s.fuel - HOVER_FUEL_KG).abs() < 1.0);
    }

    #[test]
    fn two_d_spawn_is_east_up() {
        let mut rng = SmallRng::seed_from_u64(3);
        let s = Scenario::Slam.spawn(&mut rng);
        let pad = crate::earth::pad_geodetic();
        let origin = crate::earth::pad_ecef();
        let enu = crate::earth::ecef_to_enu(s.r_eci, origin, pad.lat, pad.lon);
        assert!(
            enu.y.abs() < 2.0,
            "2D spawn should sit in the east-up plane, north {}",
            enu.y
        );
        let (east, north, up) = crate::earth::enu_basis(pad.lat, pad.lon);
        let bx = s.q_body_to_eci.rotate(Vec3::X);
        let by = s.q_body_to_eci.rotate(Vec3::Y);
        assert!(bx.dot(up) > 0.95, "body X should be up");
        assert!(by.dot(north).abs() > 0.95, "body Y should be north");
        let _ = east;
    }
}
