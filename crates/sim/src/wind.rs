//! Layered climatological wind plus Ornstein–Uhlenbeck gusts.
//!
//! Layers are a Florida-east-coast / subtropical jet caricature — westerlies
//! aloft, lighter onshore flow near the surface — not a live weather product.

use crate::earth::enu_to_ecef_vec;
use crate::math::{lerp, Vec3};
use rand::Rng;

#[derive(Clone, Debug)]
pub struct Wind {
    pub scale: f64,
    gust_enu: Vec3,
    rng_state: u64,
}

/// (alt_m, speed_m/s, direction_from_deg) — meteorological "from" convention.
const LAYERS: [(f64, f64, f64); 8] = [
    (0.0, 6.0, 90.0),     // light easterly sea breeze
    (2_000.0, 10.0, 80.0),
    (6_000.0, 18.0, 270.0),
    (10_000.0, 38.0, 260.0), // subtropical jet
    (16_000.0, 28.0, 255.0),
    (24_000.0, 14.0, 250.0),
    (40_000.0, 8.0, 240.0),
    (80_000.0, 4.0, 220.0),
];

impl Wind {
    pub fn new(seed: u64, scale: f64) -> Self {
        Self {
            scale,
            gust_enu: Vec3::ZERO,
            rng_state: seed | 1,
        }
    }

    fn next_gauss(&mut self) -> f64 {
        // Box-Muller with xorshift
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        let u1 = ((self.rng_state & 0xFFFFFF) as f64 + 1.0) / 16_777_217.0;
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        let u2 = ((self.rng_state & 0xFFFFFF) as f64 + 1.0) / 16_777_217.0;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    pub fn step(&mut self, dt: f64, rng: &mut impl Rng) {
        let _ = rng;
        // τ ≈ 4 s gust memory
        let tau = 4.0;
        let sigma = 4.0 * self.scale;
        let a = (-dt / tau).exp();
        self.gust_enu = self.gust_enu * a
            + Vec3::new(self.next_gauss(), self.next_gauss(), self.next_gauss() * 0.3)
                * (sigma * (1.0 - a * a).sqrt());
    }

    pub fn mean_enu(&self, alt: f64) -> Vec3 {
        let a = alt.max(0.0);
        let mut i = 0;
        while i + 1 < LAYERS.len() && a >= LAYERS[i + 1].0 {
            i += 1;
        }
        let (h0, s0, d0) = LAYERS[i];
        let (h1, s1, d1) = if i + 1 < LAYERS.len() {
            LAYERS[i + 1]
        } else {
            LAYERS[i]
        };
        let t = if (h1 - h0).abs() < 1.0 {
            0.0
        } else {
            ((a - h0) / (h1 - h0)).clamp(0.0, 1.0)
        };
        let speed = lerp(s0, s1, t) * self.scale;
        let dir = lerp(d0, d1, t) * std::f64::consts::PI / 180.0;
        // "from" → toward
        let toward = dir + std::f64::consts::PI;
        Vec3::new(speed * toward.sin(), speed * toward.cos(), 0.0)
    }

    pub fn velocity_ecef(&self, alt: f64, lat: f64, lon: f64) -> Vec3 {
        let enu = self.mean_enu(alt) + self.gust_enu;
        enu_to_ecef_vec(enu, lat, lon)
    }

    pub fn gust_speed(&self) -> f64 {
        self.gust_enu.norm()
    }
}
