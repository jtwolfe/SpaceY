//! 6DOF rigid-body simulation of a Falcon 9-class first stage.

use crate::atmosphere;
use crate::constants::*;
use crate::earth::{
    ecef_to_geodetic, eci_to_ecef, eci_vel_to_ecef_ground, gravity_j2, great_circle_m, pad_ecef,
    periapsis_radius,
};
use crate::guidance::{
    apply_residual, attitude_command, classify_phase, corridor_offset, corridor_violated,
    evaluate_contact, features, fuel_infeasible, ground_hit, impact_destroy, nav_from,
    nominal_controls, policy_residual, success, Nav, Phase, TermReason, N_WEIGHTS,
};
use crate::math::{Quat, Vec3};
use crate::constants::inertia_diag;
use crate::scenario::Scenario;
use crate::vehicle::{aero, check_destruction_limits, propulsion, rcs_moment, DestroyReason};
use crate::wind::{Weather, Wind};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use serde::Serialize;

fn predicted_overshoot_proxy(nav: &Nav) -> f64 {
    let vh = (nav.v_enu.x * nav.v_enu.x + nav.v_enu.y * nav.v_enu.y).sqrt();
    let t = nav.alt / (-nav.v_enu.z).max(30.0);
    vh * t * 0.55 - nav.range_h
}

#[derive(Clone, Debug)]
pub struct Sim {
    pub t: f64,
    pub r_eci: Vec3,
    pub v_eci: Vec3,
    pub q_body_to_eci: Quat,
    pub omega_body: Vec3,
    pub fuel: f64,
    pub wind: Wind,
    pub destroy_enabled: bool,
    pub wind_scale: f64,
    pub scenario: Scenario,
    pub weather: Weather,
    pub weights: Vec<f64>,
    pub start_ecef: Vec3,
    pub landing_latched: bool,
    pub deorbit_complete: bool,
    pub entry_latched: bool,
    pub min_range_gc: f64,
    pub intact: bool,
    pub destroy_reason: DestroyReason,
    pub term: TermReason,
    pub phase: Phase,
    pub last_nav: Nav,
    pub last_aero_q: f64,
    pub last_mach: f64,
    pub last_aoa: f64,
    pub last_heat: f64,
    pub last_cd: f64,
    pub last_throttle: f64,
    pub last_thrust: f64,
    pub last_n_engines: u8,
    pub last_fins: [f64; 3],
    pub last_gimbal: [f64; 2],
    pub last_accel_g: f64,
    pub last_density: f64,
    pub last_periapsis_alt: f64,
    pub seed: u32,
}

impl Sim {
    pub fn new(seed: u32, destroy_enabled: bool, wind_scale: f64) -> Self {
        Self::new_with(
            seed,
            destroy_enabled,
            wind_scale,
            Scenario::Rtls,
            Weather::default(),
        )
    }

    pub fn new_with(
        seed: u32,
        destroy_enabled: bool,
        wind_scale: f64,
        scenario: Scenario,
        weather: Weather,
    ) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed as u64 + 17);
        let spawn = scenario.spawn(&mut rng);
        let fuel = spawn.fuel;
        let mut s = Self {
            t: 0.0,
            r_eci: spawn.r_eci,
            v_eci: spawn.v_eci,
            q_body_to_eci: spawn.q_body_to_eci,
            omega_body: Vec3::ZERO,
            fuel,
            wind: Wind::new_weather(seed as u64 + 99, wind_scale, weather),
            destroy_enabled,
            wind_scale,
            scenario,
            weather,
            weights: vec![0.0; N_WEIGHTS],
            start_ecef: spawn.start_ecef,
            landing_latched: false,
            deorbit_complete: false,
            entry_latched: false,
            min_range_gc: f64::INFINITY,
            intact: true,
            destroy_reason: DestroyReason::None,
            term: TermReason::None,
            phase: Phase::Exo,
            last_nav: nav_from(
                spawn.start_ecef,
                Vec3::ZERO,
                Vec3::X,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                fuel,
                wet_mass(fuel),
            ),
            last_aero_q: 0.0,
            last_mach: 0.0,
            last_aoa: 0.0,
            last_heat: 0.0,
            last_cd: 0.8,
            last_throttle: 0.0,
            last_thrust: 0.0,
            last_n_engines: 0,
            last_fins: [0.0; 3],
            last_gimbal: [0.0; 2],
            last_accel_g: 0.0,
            last_density: 0.0,
            last_periapsis_alt: 0.0,
            seed,
        };
        s.refresh_nav();
        s.phase = classify_phase(
            &s.last_nav,
            s.landing_latched,
            s.deorbit_complete,
            s.entry_latched,
            s.scenario,
        );
        s
    }

    pub fn set_weights(&mut self, w: &[f64]) {
        self.weights = w.to_vec();
        if self.weights.len() < N_WEIGHTS {
            self.weights.resize(N_WEIGHTS, 0.0);
        }
    }

    pub fn terminated(&self) -> bool {
        self.term != TermReason::None
    }

    pub fn adaptive_dt(&self) -> f64 {
        let burning = self.last_thrust > 1_000.0;
        if burning {
            if self.last_nav.engine_alt > 2_000.0 {
                0.04
            } else {
                0.02
            }
        } else if self.last_nav.alt > 150_000.0 && self.last_aero_q < 50.0 {
            1.20
        } else if self.last_nav.alt > 65_000.0 && self.last_aero_q < 200.0 {
            0.20
        } else if self.last_aero_q > 40_000.0 {
            0.008
        } else if self.last_aero_q > 15_000.0 {
            0.012
        } else if self.last_nav.alt > 25_000.0 && self.last_aero_q < 8_000.0 {
            0.07
        } else if self.last_nav.engine_alt > 2_000.0 {
            0.04
        } else {
            0.02
        }
    }

    fn refresh_nav(&mut self) {
        let (r_ecef, v_g) = eci_vel_to_ecef_ground(self.r_eci, self.v_eci, self.t);
        let geo = ecef_to_geodetic(r_ecef);
        let body_x = self.q_body_to_eci.rotate(Vec3::X);
        // body X in ECEF
        let q_e = crate::earth::q_eci_to_ecef(self.t);
        let body_x_ecef = q_e.rotate(body_x);
        let engine_ecef = r_ecef + q_e.rotate(self.q_body_to_eci.rotate(Vec3::new(-STAGE_LENGTH_M * 0.5, 0.0, 0.0)));
        let engine_alt = ecef_to_geodetic(engine_ecef).alt;
        let peri_r = periapsis_radius(self.r_eci, self.v_eci);
        let peri_alt = if peri_r.is_finite() {
            peri_r - EARTH_RADIUS_EQ
        } else {
            f64::INFINITY
        };
        self.last_periapsis_alt = peri_alt;
        let mut nav = nav_from(
            r_ecef,
            v_g,
            body_x_ecef,
            geo.alt,
            engine_alt,
            self.last_aero_q,
            self.last_aoa,
            self.last_mach,
            self.fuel,
            wet_mass(self.fuel),
        );
        nav.periapsis_alt = peri_alt;
        if nav.range_gc < self.min_range_gc {
            self.min_range_gc = nav.range_gc;
        }
        self.last_nav = nav;
    }

    pub fn step(&mut self, dt: f64) {
        if self.terminated() {
            return;
        }
        let dt_max = if self.last_nav.alt > 140_000.0 && self.last_aero_q < 20.0 {
            1.50
        } else {
            0.50
        };
        let dt = dt.clamp(0.001, dt_max);
        let mut rng = SmallRng::seed_from_u64((self.seed as u64).wrapping_add((self.t * 1e4) as u64));
        self.wind.step(dt, &mut rng);

        let (r_ecef, v_ground) = eci_vel_to_ecef_ground(self.r_eci, self.v_eci, self.t);
        let geo = ecef_to_geodetic(r_ecef);
        let air = atmosphere::lookup_scaled(geo.alt, self.weather.density_scale());
        self.last_density = air.density;
        let q_e = crate::earth::q_eci_to_ecef(self.t);
        let v_wind = self.wind.velocity_ecef(geo.alt, geo.lat, geo.lon);
        let v_air_ecef = v_ground - v_wind;
        let v_air_eci = q_e.conjugate().rotate(v_air_ecef);
        let v_rel_body = self.q_body_to_eci.conjugate().rotate(v_air_eci);

        self.refresh_nav();
        if !self.landing_latched
            && crate::guidance::should_start_landing_for(&self.last_nav, self.scenario)
        {
            self.landing_latched = true;
        }
        if !self.deorbit_complete && self.last_nav.periapsis_alt <= DEORBIT_PERI_DONE_M {
            self.deorbit_complete = true;
        }
        self.phase = classify_phase(
            &self.last_nav,
            self.landing_latched,
            self.deorbit_complete,
            self.entry_latched,
            self.scenario,
        );
        if self.phase == Phase::Entry {
            self.entry_latched = true;
        }

        let (mut u, desired_x) = nominal_controls(&self.last_nav, self.phase, self.scenario);
        let body_x = q_e.rotate(self.q_body_to_eci.rotate(Vec3::X));
        let body_y = q_e.rotate(self.q_body_to_eci.rotate(Vec3::Y));
        let body_z = q_e.rotate(self.q_body_to_eci.rotate(Vec3::Z));
        let (gy, gz, fp, fy, fr) = attitude_command(
            body_x,
            body_y,
            body_z,
            self.omega_body,
            desired_x,
            self.phase,
            self.last_aero_q,
        );
        u.gimbal_y = gy;
        u.gimbal_z = gz;
        u.fin_pitch = fp;
        u.fin_yaw = fy;
        u.fin_roll = fr;

        let feat = features(
            &self.last_nav,
            self.phase,
            predicted_overshoot_proxy(&self.last_nav),
        );
        let resid = policy_residual(&self.weights, &feat);
        u = apply_residual(u, &resid, self.phase);

        let a = aero(v_rel_body, air, u.fin_pitch, u.fin_yaw, u.fin_roll);
        let burn = propulsion(
            u.throttle,
            u.gimbal_y,
            u.gimbal_z,
            u.n_engines,
            air.pressure_pa,
            self.fuel,
        );

        self.last_aero_q = a.q;
        self.last_mach = a.mach;
        self.last_aoa = a.aoa;
        self.last_heat = a.heat;
        self.last_cd = a.cd;
        self.last_throttle = u.throttle;
        self.last_thrust = burn.thrust;
        self.last_n_engines = u.n_engines;
        self.last_fins = [u.fin_pitch, u.fin_yaw, u.fin_roll];
        self.last_gimbal = [u.gimbal_y, u.gimbal_z];

        let mass = wet_mass(self.fuel).max(DRY_MASS_KG);
        let f_body = a.force_body + burn.force_body;
        let damp = a.q * REF_AREA_M2 * 28.0;
        let w0 = self.omega_body;
        let err = body_x.cross(desired_x.normalized());
        let err_body = Vec3::new(err.dot(body_x), err.dot(body_y), err.dot(body_z));
        let i = inertia_diag(mass);
        let m_rcs = rcs_moment(self.omega_body, err_body, i, a.q);
        let m_body = a.moment_body + burn.moment_body + m_rcs
            + Vec3::new(-damp * 1.1 * w0.x, -damp * w0.y, -damp * w0.z);
        let a_body = f_body / mass;
        let a_eci = gravity_j2(self.r_eci) + self.q_body_to_eci.rotate(a_body);
        self.last_accel_g = a_eci.norm() / G0;

        let w = self.omega_body;
        let i_w = Vec3::new(i.x * w.x, i.y * w.y, i.z * w.z);
        let mut wdot = Vec3::new(
            (m_body.x - (w.y * i_w.z - w.z * i_w.y)) / i.x.max(1.0),
            (m_body.y - (w.z * i_w.x - w.x * i_w.z)) / i.y.max(1.0),
            (m_body.z - (w.x * i_w.y - w.y * i_w.x)) / i.z.max(1.0),
        );
        // Fins + cold-gas cannot produce 100 rad/s². A stiff weathercock /
        // fin PIO under-sampled at 12–70 ms will, and then the spin check
        // fires on a numerical tumble. Cap specific torque to a physical band.
        let wdot_max = if a.q > 8_000.0 { 1.6 } else { 2.4 };
        wdot = wdot.clamp_norm(wdot_max);

        // Semi-implicit Euler — stable enough with our adaptive dt.
        self.v_eci += a_eci * dt;
        self.r_eci += self.v_eci * dt;
        self.omega_body += wdot * dt;
        self.q_body_to_eci = self.q_body_to_eci.integrate(self.omega_body, dt);
        self.fuel = (self.fuel - burn.mdot * dt).max(0.0);
        self.t += dt;

        self.refresh_nav();

        let (q_lim, g_lim, qa_lim, rate_lim) = if self.scenario.is_orbital() {
            (
                LEO_Q_DESTROY_PA,
                LEO_G_DESTROY,
                LEO_Q_ALPHA_DESTROY,
                LEO_RATE_DESTROY_RAD_S,
            )
        } else {
            (Q_DESTROY_PA, G_DESTROY, Q_ALPHA_DESTROY, RATE_DESTROY_RAD_S)
        };
        let dest = check_destruction_limits(
            self.last_aero_q,
            self.last_aoa,
            self.last_accel_g,
            self.omega_body.norm(),
            self.destroy_enabled,
            q_lim,
            g_lim,
            qa_lim,
            rate_lim,
        );
        if dest != DestroyReason::None {
            self.intact = false;
            self.destroy_reason = dest;
            self.term = TermReason::Destroyed;
            return;
        }

        if ground_hit(&self.last_nav) {
            if self.destroy_enabled && impact_destroy(&self.last_nav) {
                self.intact = false;
                self.destroy_reason = DestroyReason::GroundImpact;
                self.term = TermReason::Destroyed;
            } else {
                self.term = evaluate_contact(&self.last_nav, self.intact);
            }
            self.v_eci = Vec3::ZERO;
            self.omega_body = Vec3::ZERO;
            return;
        }

        if success(&self.last_nav, self.intact) {
            self.term = TermReason::Success;
            return;
        }

        let off = corridor_offset(eci_to_ecef(self.r_eci, self.t), self.start_ecef, pad_ecef());
        let pad_overflight = self.scenario.is_orbital() && self.min_range_gc < 25_000.0;
        if corridor_violated(&self.last_nav, off, self.scenario) && !pad_overflight {
            self.term = TermReason::Corridor;
            return;
        }
        if fuel_infeasible(&self.last_nav, self.phase) {
            self.term = TermReason::FuelInfeasible;
            return;
        }
        if self.t > self.scenario.timeout() {
            self.term = TermReason::Timeout;
        }
    }

    pub fn step_for(&mut self, seconds: f64) {
        let mut acc = 0.0;
        while acc < seconds && !self.terminated() {
            let dt = self.adaptive_dt().min(seconds - acc);
            self.step(dt);
            acc += dt;
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        let (r_ecef, v_g) = eci_vel_to_ecef_ground(self.r_eci, self.v_eci, self.t);
        let geo = ecef_to_geodetic(r_ecef);
        let pad = pad_ecef();
        let q_e = crate::earth::q_eci_to_ecef(self.t);
        let q = self.q_body_to_eci;
        Snapshot {
            t: self.t,
            lat: geo.lat,
            lon: geo.lon,
            alt: geo.alt,
            engine_alt: self.last_nav.engine_alt,
            r_ecef: r_ecef.to_array(),
            r_eci: self.r_eci.to_array(),
            v_enu: self.last_nav.v_enu.to_array(),
            pad_ecef: pad.to_array(),
            earth_angle: crate::earth::earth_angle(self.t),
            quat: [q.w, q.x, q.y, q.z],
            quat_ecef: {
                let qe = q_e.mul(q);
                [qe.w, qe.x, qe.y, qe.z]
            },
            speed: self.last_nav.speed,
            speed_inertial: self.v_eci.norm(),
            mach: self.last_mach,
            q_dyn: self.last_aero_q,
            aoa_deg: self.last_aoa * 180.0 / std::f64::consts::PI,
            fuel: self.fuel,
            fuel_frac: self.fuel / self.scenario.start_fuel(),
            mass: wet_mass(self.fuel),
            throttle: self.last_throttle,
            thrust: self.last_thrust,
            n_engines: self.last_n_engines,
            fins: self.last_fins,
            gimbal: self.last_gimbal,
            tilt_deg: self.last_nav.tilt * 180.0 / std::f64::consts::PI,
            heat: self.last_heat,
            phase: self.phase.as_str(),
            intact: self.intact,
            destroy_reason: self.destroy_reason.as_str().to_string(),
            term: self.term.as_str().to_string(),
            terminated: self.terminated(),
            success: self.term == TermReason::Success,
            range_h: self.last_nav.range_h,
            range_gc: great_circle_m(r_ecef, pad),
            pos_enu: self.last_nav.pos_enu.to_array(),
            accel_g: self.last_accel_g,
            wind_gust: self.wind.gust_speed(),
            density: self.last_density,
            density_scale: self.weather.density_scale(),
            periapsis_alt: self.last_periapsis_alt,
            scenario: self.scenario.as_str(),
            weather_storm: self.weather.storm,
            weather_shear: self.weather.shear,
            destroy_enabled: self.destroy_enabled,
            wind_scale: self.wind_scale,
            cd: self.last_cd,
            v_ground: v_g.to_array(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Snapshot {
    pub t: f64,
    pub lat: f64,
    pub lon: f64,
    pub alt: f64,
    pub engine_alt: f64,
    pub r_ecef: [f64; 3],
    pub r_eci: [f64; 3],
    pub v_enu: [f64; 3],
    pub pad_ecef: [f64; 3],
    pub earth_angle: f64,
    pub quat: [f64; 4],
    pub quat_ecef: [f64; 4],
    pub speed: f64,
    pub speed_inertial: f64,
    pub mach: f64,
    pub q_dyn: f64,
    pub aoa_deg: f64,
    pub fuel: f64,
    pub fuel_frac: f64,
    pub mass: f64,
    pub throttle: f64,
    pub thrust: f64,
    pub n_engines: u8,
    pub fins: [f64; 3],
    pub gimbal: [f64; 2],
    pub tilt_deg: f64,
    pub heat: f64,
    pub phase: &'static str,
    pub intact: bool,
    pub destroy_reason: String,
    pub term: String,
    pub terminated: bool,
    pub success: bool,
    pub range_h: f64,
    pub range_gc: f64,
    pub pos_enu: [f64; 3],
    pub accel_g: f64,
    pub wind_gust: f64,
    pub density: f64,
    pub density_scale: f64,
    pub periapsis_alt: f64,
    pub scenario: &'static str,
    pub weather_storm: bool,
    pub weather_shear: bool,
    pub destroy_enabled: bool,
    pub wind_scale: f64,
    pub cd: f64,
    pub v_ground: [f64; 3],
}

/// Fitness: higher is better. A landed booster scores ~10k; a corridor
/// violation from 80 km is a large negative.
pub fn episode_fitness(sim: &Sim) -> f64 {
    let n = &sim.last_nav;
    let mut f = 0.0;
    f += -0.004 * n.range_h;
    f += -0.003 * sim.min_range_gc.min(n.range_gc);
    f += -0.08 * n.speed;
    f += -40.0 * n.tilt;
    f += -0.0004 * n.alt;
    f += 0.02 * sim.fuel;
    match sim.term {
        TermReason::Success => {
            f += 12_000.0 - 8.0 * n.range_h - 40.0 * n.speed - 80.0 * n.tilt;
        }
        TermReason::GroundMiss => f -= 2_500.0,
        TermReason::Destroyed => f -= 4_000.0,
        TermReason::Corridor => f -= 3_000.0,
        TermReason::FuelInfeasible => f -= 3_200.0,
        TermReason::Timeout => f -= 2_800.0,
        TermReason::None => f -= 1_000.0,
    }
    if sim.destroy_reason == DestroyReason::GroundImpact {
        f -= 800.0;
    }
    f
}

pub fn run_episode(weights: &[f64], seed: u32, destroy: bool, wind_scale: f64) -> (f64, TermReason, Nav) {
    run_episode_with(
        weights,
        seed,
        destroy,
        wind_scale,
        Scenario::Rtls,
        Weather::default(),
    )
}

pub fn run_episode_with(
    weights: &[f64],
    seed: u32,
    destroy: bool,
    wind_scale: f64,
    scenario: Scenario,
    weather: Weather,
) -> (f64, TermReason, Nav) {
    let mut sim = Sim::new_with(seed, destroy, wind_scale, scenario, weather);
    sim.set_weights(weights);
    let mut guard = 0;
    let cap = if scenario.is_orbital() { 140_000 } else { 40_000 };
    while !sim.terminated() && guard < cap {
        let dt = sim.adaptive_dt();
        sim.step(dt);
        guard += 1;
    }
    if !sim.terminated() {
        sim.term = TermReason::Timeout;
    }
    (episode_fitness(&sim), sim.term, sim.last_nav)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::guidance::TermReason;
    use crate::scenario::Scenario;
    use crate::wind::Weather;

    #[test]
    fn episode_terminates() {
        let mut sim = Sim::new(1, true, 1.0);
        for _ in 0..30_000 {
            if sim.terminated() {
                break;
            }
            sim.step(sim.adaptive_dt());
        }
        assert!(sim.terminated(), "episode must terminate");
        assert!(sim.t > 5.0);
    }

    #[test]
    fn vacuum_coast_is_fast() {
        let mut sim = Sim::new(2, true, 0.0);
        sim.step(0.2);
        assert!(sim.last_nav.alt > 70_000.0);
        assert!(sim.last_nav.speed > 1_500.0);
    }

    #[test]
    fn leo_start_is_orbital_speed_in_near_vacuum() {
        let sim = Sim::new_with(3, true, 1.0, Scenario::LeoDeorbit, Weather::default());
        let snap = sim.snapshot();
        assert!(snap.alt > 200_000.0, "alt {}", snap.alt);
        assert!(
            snap.speed_inertial > 7_600.0 && snap.speed_inertial < 8_000.0,
            "inertial {}",
            snap.speed_inertial
        );
        assert!(snap.density < 1e-9, "density {}", snap.density);
        assert_eq!(snap.scenario, "leo");
        assert_eq!(sim.phase, Phase::Deorbit);
        assert!(sim.last_nav.periapsis_alt > 180_000.0);
    }

    #[test]
    fn leo_deorbit_burn_lowers_periapsis() {
        let mut sim = Sim::new_with(3, true, 0.0, Scenario::LeoDeorbit, Weather::default());
        let peri0 = sim.last_nav.periapsis_alt;
        sim.step_for(25.0);
        assert!(sim.intact);
        assert!(sim.last_nav.alt > 150_000.0);
        assert!(
            sim.last_nav.periapsis_alt < peri0 - 20_000.0
                || sim.phase == Phase::Exo,
            "peri {} → {} phase {:?}",
            peri0,
            sim.last_nav.periapsis_alt,
            sim.phase
        );
        assert!(sim.v_eci.norm() > 7_400.0);
    }

    #[test]
    fn leo_coast_does_not_corridor_trip() {
        let mut sim = Sim::new_with(3, true, 0.0, Scenario::LeoDeorbit, Weather::default());
        sim.step_for(180.0);
        assert_ne!(sim.term, TermReason::Corridor);
        assert!(sim.last_nav.alt > 140_000.0);
        assert!(sim.v_eci.norm() > 7_400.0);
        let gc = sim.snapshot().range_gc;
        assert!(gc > 10_000_000.0, "still far downrange {gc}");
    }

    #[test]
    fn default_scenario_is_rtls() {
        let sim = Sim::new(1, true, 1.0);
        assert_eq!(sim.scenario, Scenario::Rtls);
        assert!(sim.last_nav.alt < 90_000.0);
        assert!(sim.last_nav.speed < 2_500.0);
    }

    #[test]
    fn leo_interface_does_not_corridor_trip() {
        let mut sim = Sim::new_with(3, true, 0.0, Scenario::LeoDeorbit, Weather::default());
        let mut guard = 0;
        while sim.last_nav.alt > 90_000.0 && !sim.terminated() && guard < 80_000 {
            sim.step(sim.adaptive_dt());
            guard += 1;
        }
        assert_ne!(sim.term, TermReason::Corridor, "term={}", sim.term.as_str());
        assert!(sim.intact);
        assert!(sim.v_eci.norm() > 7_000.0);
        assert!(
            sim.last_nav.range_gc > 500_000.0,
            "should still be inbound, gc={}",
            sim.last_nav.range_gc
        );
    }

    #[test]
    fn leo_nominal_reaches_landing_intact() {
        // Zero residual, destruction on. The nominal must survive
        // hypersonic entry and latch a landing burn often enough that
        // CMA-ES is not staring at a wall of instant-death fitness.
        let seeds = [3u32, 20, 54, 88];
        let mut reached = 0u32;
        for seed in seeds {
            let mut sim = Sim::new_with(seed, true, 0.0, Scenario::LeoDeorbit, Weather::default());
            let mut guard = 0;
            let mut saw_land = false;
            while !sim.terminated() && guard < 140_000 {
                sim.step(sim.adaptive_dt());
                if sim.phase == Phase::Landing || sim.landing_latched {
                    saw_land = true;
                    break;
                }
                guard += 1;
            }
            if saw_land && sim.intact {
                reached += 1;
            }
        }
        assert!(
            reached >= 2,
            "expected ≥2/4 LEO nominals to reach landing intact, got {reached}"
        );
    }

    #[test]
    fn rtls_nominal_still_terminates() {
        let mut sim = Sim::new(1, true, 1.0);
        for _ in 0..30_000 {
            if sim.terminated() {
                break;
            }
            sim.step(sim.adaptive_dt());
        }
        assert!(sim.terminated());
        assert!(sim.t > 5.0);
        assert!(sim.last_nav.alt < 90_000.0);
    }
}
