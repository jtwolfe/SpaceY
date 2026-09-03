//! 6DOF rigid-body simulation of a Falcon 9-class first stage.

use crate::atmosphere;
use crate::constants::*;
use crate::earth::{
    ecef_to_enu, ecef_to_geodetic, ecef_to_eci, eci_to_ecef, eci_vel_to_ecef_ground, enu_basis,
    enu_to_ecef_vec, gravity_j2, great_circle_m, pad_ecef, pad_geodetic, periapsis_radius,
    q_eci_to_ecef,
};
use crate::guidance::{
    attitude_command, classify_phase, corridor_offset, corridor_violated, evaluate_contact,
    fuel_infeasible_for, ground_hit, impact_destroy, nav_from, nominal_controls, slam_v_ref,
    success, Controls, Nav, Phase, TermReason,
};
use crate::math::{Quat, Vec3};
use crate::policy::{
    actions_from_mlp, hidden_from_len, mlp_forward, n_weights, observe, HIDDEN_START, POLICY_DT,
};
use crate::constants::inertia_diag;
use crate::scenario::Scenario;
use crate::vehicle::{
    aero, check_destruction_limits, propulsion, rcs_commanded, rcs_moment, DestroyReason,
    EngineGate,
};
use crate::wind::{Weather, Wind};
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pilot {
    /// Hand-written tracker. Demo only — not the training prior.
    Autopilot,
    /// MLP owns throttle, gimbal, fins, cluster, and RCS. No inner PD.
    Policy,
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
    pub hidden: usize,
    pub pilot: Pilot,
    pub start_ecef: Vec3,
    pub landing_latched: bool,
    pub deorbit_complete: bool,
    pub entry_latched: bool,
    pub min_range_gc: f64,
    pub max_corridor_offset: f64,
    pub corridor_violated_once: bool,
    /// Trainer rollouts keep flying after a corridor breach so CMA-ES
    /// can rank "how far / how fast" instead of a single cliff.
    pub soft_corridor: bool,
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
    pub last_fin_delta: [f64; 4],
    pub last_gimbal: [f64; 2],
    pub last_rcs: f64,
    pub last_accel_g: f64,
    pub last_density: f64,
    pub last_periapsis_alt: f64,
    pub max_rate: f64,
    pub max_aoa: f64,
    pub max_tilt: f64,
    pub tumble_s: f64,
    pub seed: u32,
    held_controls: Controls,
    last_policy_t: f64,
    pub shaping: f64,
    pub v_slam: f64,
    pub engine: EngineGate,
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
        Self::new_with_opts(
            seed,
            destroy_enabled,
            wind_scale,
            scenario,
            weather,
            false,
        )
    }

    pub fn new_with_opts(
        seed: u32,
        destroy_enabled: bool,
        wind_scale: f64,
        scenario: Scenario,
        weather: Weather,
        hard_pad: bool,
    ) -> Self {
        // StdRng (ChaCha) is portable across wasm32 (32-bit) and native
        // x86_64. SmallRng is xoshiro256++ vs xoshiro128++.
        let mut rng = StdRng::seed_from_u64(seed as u64 + 17);
        let spawn = scenario.spawn_var(&mut rng, hard_pad);
        let fuel = spawn.fuel;
        let mut s = Self {
            t: 0.0,
            r_eci: spawn.r_eci,
            v_eci: spawn.v_eci,
            q_body_to_eci: spawn.q_body_to_eci,
            omega_body: spawn.omega_body,
            fuel,
            wind: Wind::new_weather(seed as u64 + 99, wind_scale, weather),
            destroy_enabled,
            wind_scale,
            scenario,
            weather,
            weights: vec![0.0; n_weights(HIDDEN_START)],
            hidden: HIDDEN_START,
            pilot: Pilot::Autopilot,
            start_ecef: spawn.start_ecef,
            landing_latched: false,
            deorbit_complete: false,
            entry_latched: false,
            min_range_gc: f64::INFINITY,
            max_corridor_offset: 0.0,
            corridor_violated_once: false,
            soft_corridor: false,
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
            last_fin_delta: [0.0; 4],
            last_gimbal: [0.0; 2],
            last_rcs: 0.0,
            last_accel_g: 0.0,
            last_density: 0.0,
            last_periapsis_alt: 0.0,
            max_rate: 0.0,
            max_aoa: 0.0,
            max_tilt: 0.0,
            tumble_s: 0.0,
            seed,
            held_controls: Controls::default(),
            last_policy_t: -1.0,
            shaping: 0.0,
            v_slam: 0.0,
            engine: EngineGate::default(),
        };
        s.refresh_nav();
        s.v_slam = slam_v_ref(s.last_nav.engine_alt, wet_mass(fuel));
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
        self.hidden = hidden_from_len(w.len());
        self.weights = w.to_vec();
        let n = n_weights(self.hidden);
        if self.weights.len() < n {
            self.weights.resize(n, 0.0);
        }
    }

    pub fn set_pilot(&mut self, pilot: Pilot) {
        self.pilot = pilot;
    }

    pub fn terminated(&self) -> bool {
        self.term != TermReason::None
    }

    pub fn adaptive_dt(&self) -> f64 {
        // Altitude + burn only. A Q-threshold dt used to flip between
        // host and wasm32 on a 0.1% Q difference.
        let burning = self.last_thrust > 1_000.0;
        let alt = self.last_nav.alt;
        if burning {
            if self.last_nav.engine_alt > 2_000.0 {
                0.04
            } else {
                0.02
            }
        } else if alt > 150_000.0 {
            1.20
        } else if alt > 80_000.0 {
            0.20
        } else if alt > 40_000.0 {
            0.010
        } else if alt > 12_000.0 {
            0.012
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
        let mut rng = StdRng::seed_from_u64((self.seed as u64).wrapping_add((self.t * 1e4) as u64));
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

        let body_x = q_e.rotate(self.q_body_to_eci.rotate(Vec3::X));
        let body_y = q_e.rotate(self.q_body_to_eci.rotate(Vec3::Y));
        let body_z = q_e.rotate(self.q_body_to_eci.rotate(Vec3::Z));

        let (mut u, desired_x) = match self.pilot {
            Pilot::Autopilot => {
                let (mut u, desired_x) =
                    nominal_controls(&self.last_nav, self.phase, self.scenario);
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
                (u, desired_x)
            }
            Pilot::Policy => {
                let need = self.last_policy_t < 0.0
                    || self.t - self.last_policy_t >= POLICY_DT - 1e-9;
                if need {
                    let feat = observe(&self.last_nav, body_y, self.omega_body);
                    let y = mlp_forward(&self.weights, self.hidden, &feat);
                    self.held_controls = actions_from_mlp(&y, self.scenario.plane_lock());
                    self.last_policy_t = self.t;
                }
                (self.held_controls, body_x)
            }
        };

        let (thr, n_eng) = self.engine.apply(self.t, u.throttle, u.n_engines);
        u.throttle = thr;
        u.n_engines = n_eng;
        u.gimbal_y = slew_axis(self.last_gimbal[0], u.gimbal_y, GIMBAL_SLEW_RAD_S, dt);
        u.gimbal_z = slew_axis(self.last_gimbal[1], u.gimbal_z, GIMBAL_SLEW_RAD_S, dt);
        u.fin_pitch = slew_axis(self.last_fins[0], u.fin_pitch, FIN_SLEW_RAD_S, dt);
        u.fin_yaw = slew_axis(self.last_fins[1], u.fin_yaw, FIN_SLEW_RAD_S, dt);
        u.fin_roll = slew_axis(self.last_fins[2], u.fin_roll, FIN_SLEW_RAD_S, dt);

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
        self.last_fin_delta = a.fin_delta;
        self.last_gimbal = [u.gimbal_y, u.gimbal_z];

        let mass = wet_mass(self.fuel).max(DRY_MASS_KG);
        let f_body = a.force_body + burn.force_body;
        let damp = a.q * REF_AREA_M2 * 28.0;
        let w0 = self.omega_body;
        let err = body_x.cross(desired_x.normalized());
        let err_body = Vec3::new(err.dot(body_x), err.dot(body_y), err.dot(body_z));
        let i = inertia_diag(mass);
        let m_rcs = if self.pilot == Pilot::Policy {
            rcs_commanded(Vec3::new(u.rcs_x, u.rcs_y, u.rcs_z), i, a.q)
        } else {
            rcs_moment(self.omega_body, err_body, i, a.q)
        };
        self.last_rcs = m_rcs.norm() / (i.y * RCS_ANG_ACCEL).max(1.0);
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

        if self.scenario.plane_lock() {
            self.apply_plane_lock();
        }

        self.refresh_nav();
        let mass_now = wet_mass(self.fuel).max(DRY_MASS_KG);
        self.v_slam = slam_v_ref(self.last_nav.engine_alt, mass_now);
        self.shaping += step_shaping(
            &self.last_nav,
            self.omega_body.norm(),
            burn.mdot,
            dt,
        );
        self.max_rate = self.max_rate.max(self.omega_body.norm());
        self.max_aoa = self.max_aoa.max(self.last_aoa.abs());
        self.max_tilt = self.max_tilt.max(self.last_nav.tilt);
        if self.last_aoa.abs() > 25.0 * std::f64::consts::PI / 180.0 && self.last_aero_q > 2_000.0 {
            self.tumble_s += dt;
        }

        let (q_lim, g_lim, qa_lim, rate_lim) =
            (Q_DESTROY_PA, G_DESTROY, Q_ALPHA_DESTROY, RATE_DESTROY_RAD_S);
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
            // Success box first — a 9 m/s pad sit-down is not a breakup.
            if success(&self.last_nav, self.intact) {
                self.term = TermReason::Success;
            } else if self.destroy_enabled && impact_destroy(&self.last_nav) {
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

        let off = corridor_offset(eci_to_ecef(self.r_eci, self.t), self.start_ecef, pad_ecef());
        if off > self.max_corridor_offset {
            self.max_corridor_offset = off;
        }
        if corridor_violated(&self.last_nav, off, self.scenario) {
            self.corridor_violated_once = true;
            if !self.soft_corridor {
                self.term = TermReason::Corridor;
                return;
            }
        }
        if fuel_infeasible_for(&self.last_nav, self.phase, self.scenario) {
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

    fn apply_plane_lock(&mut self) {
        let pad_g = pad_geodetic();
        let pad = pad_ecef();
        let (east, north, up) = enu_basis(pad_g.lat, pad_g.lon);
        let (r_ecef, v_g) = eci_vel_to_ecef_ground(self.r_eci, self.v_eci, self.t);
        let pos = ecef_to_enu(r_ecef, pad, pad_g.lat, pad_g.lon);
        let v_enu = Vec3::new(v_g.dot(east), v_g.dot(north), v_g.dot(up));
        let pos2 = Vec3::new(pos.x, 0.0, pos.z);
        let v2 = Vec3::new(v_enu.x, 0.0, v_enu.z);
        let r2 = pad + enu_to_ecef_vec(pos2, pad_g.lat, pad_g.lon);
        let v_ground = enu_to_ecef_vec(v2, pad_g.lat, pad_g.lon);
        let q_e = q_eci_to_ecef(self.t);
        let omega_e = Vec3::new(0.0, 0.0, EARTH_OMEGA);
        self.r_eci = ecef_to_eci(r2, self.t);
        self.v_eci = q_e.conjugate().rotate(v_ground + omega_e.cross(r2));

        let omega_ecef = q_e.rotate(self.q_body_to_eci.rotate(self.omega_body));
        let pitch = omega_ecef.dot(north);

        let bx_ecef = q_e.rotate(self.q_body_to_eci.rotate(Vec3::X));
        let bx_enu = Vec3::new(bx_ecef.dot(east), bx_ecef.dot(north), bx_ecef.dot(up));
        let mut bx_plane = Vec3::new(bx_enu.x, 0.0, bx_enu.z);
        if bx_plane.norm() < 1e-6 {
            bx_plane = Vec3::new(0.0, 0.0, 1.0);
        } else {
            bx_plane = bx_plane.normalized();
        }
        let bx_w = enu_to_ecef_vec(bx_plane, pad_g.lat, pad_g.lon);
        let bz_w = bx_w.cross(north);
        let bz_w = if bz_w.norm() < 1e-8 {
            east
        } else {
            bz_w.normalized()
        };
        let q_body_ecef = Quat::from_axes(bx_w, north, bz_w);
        self.q_body_to_eci = q_e.conjugate().mul(q_body_ecef).normalized();
        self.omega_body = self
            .q_body_to_eci
            .conjugate()
            .rotate(q_e.conjugate().rotate(north * pitch));
    }

    fn slam_curve_xyz(&self) -> Vec<f32> {
        const N: usize = 20;
        let pad_g = pad_geodetic();
        let pad = pad_ecef();
        let (r_ecef, _) = eci_vel_to_ecef_ground(self.r_eci, self.v_eci, self.t);
        let q_e = q_eci_to_ecef(self.t);
        let engine_ecef = r_ecef
            + q_e.rotate(
                self.q_body_to_eci
                    .rotate(Vec3::new(-STAGE_LENGTH_M * 0.5, 0.0, 0.0)),
            );
        let eng = ecef_to_enu(engine_ecef, pad, pad_g.lat, pad_g.lon);
        let mut out = Vec::with_capacity(N * 3);
        for i in 0..N {
            let s = i as f64 / (N - 1) as f64;
            let e = eng.x * (1.0 - s);
            let n = if self.scenario.plane_lock() {
                0.0
            } else {
                eng.y * (1.0 - s)
            };
            let u = eng.z * (1.0 - s);
            let p = pad + enu_to_ecef_vec(Vec3::new(e, n, u), pad_g.lat, pad_g.lon);
            out.push(p.x as f32);
            out.push(p.y as f32);
            out.push(p.z as f32);
        }
        out
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
            fin_delta: self.last_fin_delta,
            gimbal: self.last_gimbal,
            rcs: self.last_rcs,
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
            weather_dir_off: self.weather.dir_off_deg,
            destroy_enabled: self.destroy_enabled,
            wind_scale: self.wind_scale,
            cd: self.last_cd,
            v_ground: v_g.to_array(),
            pilot: match self.pilot {
                Pilot::Autopilot => "autopilot",
                Pilot::Policy => "policy",
            },
            v_slam: self.v_slam,
            slam_xyz: self.slam_curve_xyz(),
            plane_lock: self.scenario.plane_lock(),
            lights: self.engine.lights,
            relights: self.engine.relights,
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
    pub fin_delta: [f64; 4],
    pub gimbal: [f64; 2],
    pub rcs: f64,
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
    pub weather_dir_off: f64,
    pub destroy_enabled: bool,
    pub wind_scale: f64,
    pub cd: f64,
    pub v_ground: [f64; 3],
    pub pilot: &'static str,
    pub v_slam: f64,
    pub slam_xyz: Vec<f32>,
    pub plane_lock: bool,
    pub lights: u32,
    pub relights: u32,
}

fn slew_axis(current: f64, target: f64, rate: f64, dt: f64) -> f64 {
    let max = (rate * dt).max(0.0);
    current + (target - current).clamp(-max, max)
}

fn step_shaping(nav: &Nav, rate: f64, mdot: f64, dt: f64) -> f64 {
    let live = 0.35 * dt;
    let climb = nav.v_enu.z.max(0.0).min(80.0);
    let path = -(0.025 * nav.range_h.min(800.0)
        + 0.04 * nav.speed.min(160.0)
        + 0.16 * climb
        + 6.0 * nav.tilt.min(1.2)
        + 3.0 * rate.min(2.5))
        * dt;
    let fuel = -0.00015 * mdot * dt;
    live + path + fuel
}

#[cfg(test)]
fn hop_fitness(sim: &Sim) -> f64 {
    episode_fitness(sim)
}

/// Fitness: higher is better. One kernel for hops and glide/RTLS so promote
/// is not a landscape teleport. Hop stages only differ by a shorter timeout.
/// A landing still dwarfs every miss.
pub fn episode_fitness(sim: &Sim) -> f64 {
    let n = &sim.last_nav;
    let mut f = sim.shaping;
    let range = n.range_h;
    let closest = sim.min_range_gc.min(n.range_gc).min(n.range_h);
    // Dense landing terms (capped so high-energy starts saturate until slow).
    f -= 0.20 * n.engine_alt.min(400.0);
    f -= 12.0 * n.speed.min(80.0);
    f -= 50.0 * n.tilt.min(1.6);
    f -= 4.0 * n.range_h.min(250.0);
    f -= 20.0 * sim.max_rate.min(3.0);
    // Hops: tilt at contact. Peak tilt mid-burn made a 6DOF suicide burn
    // lose to a fins-straight dart. Glide/RTLS still pay peak (entry cartwheel).
    let tilt_fee = if sim.scenario.is_terminal_hop() {
        n.tilt
    } else {
        sim.max_tilt
    };
    f -= 2_000.0 * (tilt_fee - 0.20).max(0.0).min(1.5);
    // Long-flight terms (tiny on a pad hop).
    f -= 0.022 * range.min(120_000.0);
    f -= 0.018 * closest.min(120_000.0);
    f -= 0.045 * sim.max_corridor_offset.min(40_000.0);
    f -= 0.055 * n.speed.min(8_000.0);
    f += 0.015 * sim.fuel.min(80_000.0);
    f -= 260.0 * (sim.max_rate.min(6.0) - 3.0).max(0.0);
    f -= 220.0 * sim.max_aoa.min(1.8);
    f -= 40.0 * sim.tumble_s.min(60.0);
    if n.alt < 45_000.0 {
        f -= 0.05 * n.speed.min(3_000.0) * (1.0 - n.alt / 45_000.0);
    }
    if sim.max_aoa > 0.60 && sim.term != TermReason::Success {
        f -= 1_400.0;
    }
    if sim.max_rate > 1.2 && sim.term != TermReason::Success {
        f -= 900.0;
    }
    if sim.corridor_violated_once {
        f -= 2_200.0;
    }
    match sim.term {
        TermReason::Success => {
            f += 14_000.0 - 10.0 * n.range_h - 50.0 * n.speed - 90.0 * n.tilt;
        }
        TermReason::GroundMiss => f -= 1_800.0,
        TermReason::Destroyed => f -= 3_600.0,
        TermReason::Corridor => f -= 2_800.0,
        TermReason::FuelInfeasible => f -= 2_400.0,
        TermReason::Timeout => f -= 2_200.0,
        TermReason::None => f -= 800.0,
    }
    if sim.destroy_reason == DestroyReason::GroundImpact {
        f -= 700.0;
    }
    // Airborne non-contact misses (fuel-out, hang, in-air breakup, corridor)
    // must not beat a slap. Contact outcomes have engine_alt < 0.4.
    if sim.term != TermReason::Success && n.engine_alt > 0.4 {
        f -= 4_200.0 + 6.0 * n.engine_alt.min(800.0);
    }
    // Hops: a fast upright slap must lose to a slower, slightly messier burn.
    // Cap so a 42 m/s dart still beats a 400 m hang.
    if sim.scenario.is_terminal_hop()
        && sim.term != TermReason::Success
        && n.engine_alt < 0.4
    {
        f -= 80.0 * (n.speed - SUCCESS_SPEED_MPS).max(0.0).min(70.0);
    }
    f - relight_penalty(sim)
}

fn relight_penalty(sim: &Sim) -> f64 {
    RELIGHT_FITNESS * (sim.engine.relights as f64).min(12.0)
}

const PATH_MAX: usize = 88;

/// Downsampled ECEF / geodetic trail from one training rollout.
#[derive(Clone, Debug)]
pub struct EpisodeTrace {
    pub fitness: f64,
    pub term: TermReason,
    pub success: bool,
    pub t_end: f32,
    pub xyz: Vec<f32>,
    pub lat: Vec<f32>,
    pub lon: Vec<f32>,
    pub nav: Nav,
}

fn push_path_sample(sim: &Sim, xyz: &mut Vec<f32>, lat: &mut Vec<f32>, lon: &mut Vec<f32>) {
    let r = eci_to_ecef(sim.r_eci, sim.t);
    let g = ecef_to_geodetic(r);
    if xyz.len() / 3 >= PATH_MAX {
        let i = xyz.len() - 3;
        xyz[i] = r.x as f32;
        xyz[i + 1] = r.y as f32;
        xyz[i + 2] = r.z as f32;
        let n = lat.len() - 1;
        lat[n] = g.lat as f32;
        lon[n] = g.lon as f32;
        return;
    }
    xyz.push(r.x as f32);
    xyz.push(r.y as f32);
    xyz.push(r.z as f32);
    lat.push(g.lat as f32);
    lon.push(g.lon as f32);
}

/// Same adaptive_dt loop the browser `fast_forward` / display warp uses.
pub fn run_hover_policy_snapshot(
    seed: u32,
    destroy: bool,
    wind_scale: f64,
    max_steps: u32,
) -> Snapshot {
    let mut sim = Sim::new_with(seed, destroy, wind_scale, Scenario::Pad, Weather::default());
    sim.pilot = Pilot::Policy;
    sim.set_weights(&vec![0.0; crate::policy::n_weights(crate::policy::HIDDEN_START)]);
    let cap = if max_steps == 0 { 20_000 } else { max_steps };
    let mut guard = 0u32;
    while !sim.terminated() && guard < cap {
        sim.step(sim.adaptive_dt());
        guard += 1;
    }
    sim.snapshot()
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
    let tr = run_episode_traced(
        weights,
        seed,
        destroy,
        wind_scale,
        scenario,
        weather,
        false,
        false,
    );
    (tr.fitness, tr.term, tr.nav)
}

/// Full training rollout: soft corridor, downsampled trail for the swarm view.
pub fn run_episode_traced(
    weights: &[f64],
    seed: u32,
    destroy: bool,
    wind_scale: f64,
    scenario: Scenario,
    weather: Weather,
    soft_corridor: bool,
    hard_pad: bool,
) -> EpisodeTrace {
    let mut sim = Sim::new_with_opts(seed, destroy, wind_scale, scenario, weather, hard_pad);
    sim.set_weights(weights);
    sim.pilot = Pilot::Policy;
    sim.soft_corridor = soft_corridor;
    let mut xyz = Vec::with_capacity(PATH_MAX * 3);
    let mut lat = Vec::with_capacity(PATH_MAX);
    let mut lon = Vec::with_capacity(PATH_MAX);
    push_path_sample(&sim, &mut xyz, &mut lat, &mut lon);
    let mut last_rec_t = sim.t;
    let mut last_r = eci_to_ecef(sim.r_eci, sim.t);
    let mut guard = 0;
    let cap = 40_000;
    while !sim.terminated() && guard < cap {
        let dt = sim.adaptive_dt();
        sim.step(dt);
        guard += 1;
        let r = eci_to_ecef(sim.r_eci, sim.t);
        if sim.t - last_rec_t >= 0.55 || (r - last_r).norm() > 2_800.0 {
            push_path_sample(&sim, &mut xyz, &mut lat, &mut lon);
            last_rec_t = sim.t;
            last_r = r;
        }
    }
    if !sim.terminated() {
        sim.term = TermReason::Timeout;
    }
    push_path_sample(&sim, &mut xyz, &mut lat, &mut lon);
    EpisodeTrace {
        fitness: episode_fitness(&sim),
        term: sim.term,
        success: sim.term == TermReason::Success,
        t_end: sim.t as f32,
        xyz,
        lat,
        lon,
        nav: sim.last_nav,
    }
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
    fn default_scenario_is_rtls() {
        let sim = Sim::new(1, true, 1.0);
        assert_eq!(sim.scenario, Scenario::Rtls);
        assert!(sim.last_nav.alt < 90_000.0);
        assert!(sim.last_nav.speed < 2_500.0);
    }

    #[test]
    fn pad_policy_zero_weights_does_not_land() {
        let mut sim = Sim::new_with(1, true, 0.0, Scenario::Pad, Weather::default());
        sim.pilot = Pilot::Policy;
        sim.set_weights(&vec![0.0; crate::policy::n_weights(crate::policy::HIDDEN_START)]);
        let mut guard = 0;
        while !sim.terminated() && guard < 20_000 {
            sim.step(sim.adaptive_dt());
            guard += 1;
        }
        assert!(sim.terminated());
        assert_ne!(sim.term, TermReason::Success);
        assert_eq!(sim.last_n_engines, 0);
        assert_eq!(sim.engine.lights, 0);
        assert!(sim.last_rcs.abs() < 1e-9);
        assert_eq!(sim.snapshot().scenario, "pad");
    }

    #[test]
    fn glide_zero_policy_does_not_land() {
        let mut sim = Sim::new_with(2, true, 0.0, Scenario::Glide, Weather::default());
        sim.pilot = Pilot::Policy;
        sim.set_weights(&vec![0.0; crate::policy::n_weights(crate::policy::HIDDEN_START)]);
        let mut guard = 0;
        while !sim.terminated() && guard < 40_000 {
            sim.step(sim.adaptive_dt());
            guard += 1;
        }
        assert!(sim.terminated());
        assert_ne!(sim.term, TermReason::Success);
        assert_eq!(sim.engine.lights, 0);
        assert!(sim.last_rcs.abs() < 1e-9);
    }

    fn throttle_out_bias(hidden: usize) -> usize {
        let b1 = hidden * crate::policy::N_IN;
        let w2 = b1 + hidden;
        w2 + crate::policy::N_OUT * hidden
    }

    fn pad_rollout(seed: u32, set: impl Fn(&mut [f64])) -> (f64, TermReason, f64, f64, u32) {
        let mut sim = Sim::new_with(seed, true, 0.0, Scenario::Pad, Weather::default());
        sim.pilot = Pilot::Policy;
        let h = crate::policy::HIDDEN_START;
        let mut w = vec![0.0; crate::policy::n_weights(h)];
        set(&mut w);
        sim.set_weights(&w);
        let mut guard = 0;
        while !sim.terminated() && guard < 20_000 {
            sim.step(sim.adaptive_dt());
            guard += 1;
        }
        (
            episode_fitness(&sim),
            sim.term,
            sim.last_nav.tilt.to_degrees(),
            sim.max_rate.to_degrees(),
            sim.engine.lights,
        )
    }

    #[test]
    fn plane_lock_rcs_does_not_cartwheel() {
        let b2 = throttle_out_bias(crate::policy::HIDDEN_START);
        let rcs = pad_rollout(7, |w| w[b2 + 8] = 2.0);
        assert_eq!(rcs.4, 0);
        assert!(
            rcs.2.abs() < 8.0,
            "2D pad RCS should be off, tilt {}°",
            rcs.2
        );
    }

    #[test]
    fn gimbaled_pad_burn_loses_to_upright_fall() {
        let b2 = throttle_out_bias(crate::policy::HIDDEN_START);
        let fall = pad_rollout(7, |_| {});
        let gimb = pad_rollout(7, |w| {
            w[b2] = 2.0;
            w[b2 + 1] = 2.0;
        });
        assert!(
            gimb.0 < fall.0,
            "TVC cartwheel {} should lose to upright slap {}",
            gimb.0,
            fall.0
        );
    }

    #[test]
    fn policy_cannot_pwm_faster_than_min_burn() {
        let mut sim = Sim::new_with(3, true, 0.0, Scenario::Pad, Weather::default());
        sim.pilot = Pilot::Policy;
        let h = crate::policy::HIDDEN_START;
        let n = crate::policy::n_weights(h);
        let b2 = throttle_out_bias(h);
        let mut on = vec![0.0; n];
        on[b2] = 2.0;
        let off = vec![0.0; n];
        let mut cmd_on = true;
        let mut next_flip = 0.0;
        while sim.t < 2.0 && !sim.terminated() {
            if sim.t + 1e-12 >= next_flip {
                sim.set_weights(if cmd_on { &on } else { &off });
                cmd_on = !cmd_on;
                next_flip += crate::policy::POLICY_DT;
            }
            sim.step(sim.adaptive_dt());
        }
        assert_eq!(sim.engine.lights, 1);
        assert_eq!(sim.engine.relights, 0);
        assert!(sim.last_throttle >= THROTTLE_MIN * 0.99);
    }

    #[test]
    fn relights_cut_fitness() {
        let mut a = Sim::new_with(1, true, 0.0, Scenario::Pad, Weather::default());
        a.term = TermReason::Success;
        let mut b = a.clone();
        b.engine.relights = 4;
        let da = hop_fitness(&a) - hop_fitness(&b);
        assert!(
            (da - RELIGHT_FITNESS * 4.0).abs() < 1e-9,
            "relight delta {da}"
        );
        a.scenario = Scenario::Glide;
        b.scenario = Scenario::Glide;
        let dg = episode_fitness(&a) - episode_fitness(&b);
        assert!((dg - RELIGHT_FITNESS * 4.0).abs() < 1e-9, "glide delta {dg}");
    }

    #[test]
    fn hop_and_glide_share_the_fitness_kernel() {
        let mut hop = Sim::new_with(1, true, 0.0, Scenario::Pad, Weather::default());
        hop.term = TermReason::Timeout;
        let mut glide = hop.clone();
        glide.scenario = Scenario::Glide;
        let dh = episode_fitness(&hop);
        let dg = episode_fitness(&glide);
        assert!(
            (dh - dg).abs() < 1e-9,
            "scenario label must not change the kernel: hop={dh} glide={dg}"
        );
    }

    #[test]
    fn climbing_away_loses_to_a_pad_slap() {
        let mut fly = Sim::new_with(1, true, 0.0, Scenario::Pad, Weather::default());
        fly.term = TermReason::Timeout;
        fly.last_nav.engine_alt = 400.0;
        fly.last_nav.alt = 430.0;
        fly.last_nav.speed = 6.0;
        fly.last_nav.v_enu = Vec3::new(0.0, 0.0, 6.0);
        fly.last_nav.tilt = 0.05;
        let mut slap = fly.clone();
        slap.term = TermReason::Destroyed;
        slap.destroy_reason = DestroyReason::GroundImpact;
        slap.last_nav.engine_alt = 0.2;
        slap.last_nav.alt = 25.0;
        slap.last_nav.speed = 42.0;
        slap.last_nav.v_enu = Vec3::new(0.0, 0.0, -42.0);
        let up = episode_fitness(&fly);
        let hit = episode_fitness(&slap);
        assert!(
            up < hit,
            "timeout-high {up} should lose to ground impact {hit}"
        );
    }

    fn hop_base() -> Sim {
        Sim::new_with(1, true, 0.0, Scenario::Attitude, Weather::default())
    }

    fn airborne_at(mut sim: Sim, term: TermReason, alt: f64, speed: f64) -> Sim {
        sim.term = term;
        sim.last_nav.engine_alt = alt;
        sim.last_nav.alt = alt + 30.0;
        sim.last_nav.speed = speed;
        sim.last_nav.v_enu = Vec3::new(0.0, 0.0, speed);
        sim.last_nav.tilt = 0.05;
        sim.last_nav.fuel = 20.0;
        sim.fuel = 20.0;
        sim
    }

    fn ground_slap(mut sim: Sim) -> Sim {
        sim.term = TermReason::Destroyed;
        sim.destroy_reason = DestroyReason::GroundImpact;
        sim.last_nav.engine_alt = 0.2;
        sim.last_nav.alt = 25.0;
        sim.last_nav.speed = 42.0;
        sim.last_nav.v_enu = Vec3::new(0.0, 0.0, -42.0);
        sim.last_nav.tilt = 0.05;
        sim
    }

    fn ground_miss(mut sim: Sim) -> Sim {
        sim.term = TermReason::GroundMiss;
        sim.destroy_reason = DestroyReason::None;
        sim.last_nav.engine_alt = 0.2;
        sim.last_nav.alt = 25.0;
        sim.last_nav.speed = 12.0;
        sim.last_nav.v_enu = Vec3::new(0.0, 0.0, -12.0);
        sim.last_nav.tilt = 0.08;
        sim
    }

    fn cartwheel_contact(mut sim: Sim) -> Sim {
        sim.term = TermReason::Destroyed;
        sim.destroy_reason = DestroyReason::Spin;
        sim.last_nav.engine_alt = 0.2;
        sim.last_nav.alt = 25.0;
        sim.last_nav.speed = 42.0;
        sim.last_nav.v_enu = Vec3::new(0.0, 0.0, -42.0);
        sim.last_nav.tilt = 0.80;
        sim.max_tilt = 0.80;
        sim
    }

    fn slower_destroy_with_peak_tilt(mut sim: Sim) -> Sim {
        sim.term = TermReason::Destroyed;
        sim.destroy_reason = DestroyReason::GroundImpact;
        sim.last_nav.engine_alt = 0.2;
        sim.last_nav.alt = 25.0;
        sim.last_nav.speed = 25.0;
        sim.last_nav.v_enu = Vec3::new(0.0, 0.0, -25.0);
        sim.last_nav.tilt = 0.05;
        sim.max_tilt = 0.70;
        sim
    }

    #[test]
    fn fuel_out_high_loses_to_a_slap() {
        let climb = airborne_at(
            hop_base(),
            TermReason::FuelInfeasible,
            400.0,
            40.0,
        );
        let slap = ground_slap(hop_base());
        let up = episode_fitness(&climb);
        let hit = episode_fitness(&slap);
        assert!(
            up < hit,
            "fuel-out-high {up} should lose to slap {hit}"
        );
    }

    #[test]
    fn airborne_spin_loses_to_a_slap() {
        let mut spin = airborne_at(hop_base(), TermReason::Destroyed, 400.0, 40.0);
        spin.destroy_reason = DestroyReason::Spin;
        let slap = ground_slap(hop_base());
        let air = episode_fitness(&spin);
        let hit = episode_fitness(&slap);
        assert!(
            air < hit,
            "airborne spin {air} should lose to slap {hit}"
        );
    }

    #[test]
    fn ground_miss_beats_airborne_fuel_out() {
        let miss = ground_miss(hop_base());
        let climb = airborne_at(
            hop_base(),
            TermReason::FuelInfeasible,
            400.0,
            40.0,
        );
        let sit = episode_fitness(&miss);
        let up = episode_fitness(&climb);
        assert!(
            sit > up,
            "sit-down miss {sit} should beat fuel-out-high {up}"
        );
    }

    #[test]
    fn ground_miss_beats_a_dart() {
        let sit = episode_fitness(&ground_miss(hop_base()));
        let dart = episode_fitness(&ground_slap(hop_base()));
        assert!(sit > dart, "sit-down {sit} should beat dart {dart}");
    }

    #[test]
    fn dart_beats_hang_timeout() {
        let hang = airborne_at(hop_base(), TermReason::Timeout, 400.0, 6.0);
        let dart = ground_slap(hop_base());
        let up = episode_fitness(&hang);
        let hit = episode_fitness(&dart);
        assert!(up < hit, "hang-timeout {up} should lose to dart {hit}");
    }

    #[test]
    fn cartwheel_contact_loses_to_a_dart() {
        let cart = episode_fitness(&cartwheel_contact(hop_base()));
        let dart = episode_fitness(&ground_slap(hop_base()));
        assert!(
            cart < dart,
            "cartwheel contact {cart} should lose to dart {dart}"
        );
    }

    #[test]
    fn peak_tilt_sitdown_still_beats_a_dart() {
        let mut sit = ground_miss(hop_base());
        sit.max_tilt = 0.70;
        let slow = episode_fitness(&sit);
        let dart = episode_fitness(&ground_slap(hop_base()));
        assert!(
            slow > dart,
            "sit-down with mid-burn tilt {slow} should still beat dart {dart}"
        );
    }

    #[test]
    fn slower_destroy_with_peak_tilt_beats_a_dart() {
        let burn = episode_fitness(&slower_destroy_with_peak_tilt(hop_base()));
        let dart = episode_fitness(&ground_slap(hop_base()));
        assert!(
            burn > dart,
            "slower destroy with peak tilt {burn} should beat dart {dart}"
        );
    }

    #[test]
    fn two_d_policy_stays_in_pitch_plane() {
        let mut sim = Sim::new_with(4, true, 0.0, Scenario::Pad, Weather::default());
        sim.pilot = Pilot::Policy;
        let mut w = vec![0.0; crate::policy::n_weights(crate::policy::HIDDEN_START)];
        w[0] = 0.8;
        sim.set_weights(&w);
        for _ in 0..80 {
            if sim.terminated() {
                break;
            }
            sim.step(sim.adaptive_dt());
        }
        assert!(
            sim.last_nav.pos_enu.y.abs() < 1.5,
            "north leak {}",
            sim.last_nav.pos_enu.y
        );
        assert!(
            sim.last_nav.v_enu.y.abs() < 0.8,
            "north vel {}",
            sim.last_nav.v_enu.y
        );
        assert!(sim.last_gimbal[1].abs() < 1e-9);
    }

    #[test]
    fn pad_autopilot_reaches_the_pad_theater() {
        let mut sim = Sim::new_with(2, true, 0.0, Scenario::Pad, Weather::default());
        sim.pilot = Pilot::Autopilot;
        let mut guard = 0;
        while !sim.terminated() && guard < 20_000 {
            sim.step(sim.adaptive_dt());
            guard += 1;
        }
        assert!(sim.terminated());
        assert!(sim.last_nav.engine_alt < 150.0);
        assert!(
            sim.last_nav.range_h < 2_000.0,
            "hovered away from pad: range {}",
            sim.last_nav.range_h
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
