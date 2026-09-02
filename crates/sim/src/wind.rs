//! Layered climatological wind plus Ornstein–Uhlenbeck gusts.
//!
//! Layers are a Florida-east-coast / subtropical jet caricature — westerlies
//! aloft, lighter onshore flow near the surface — not a live weather product.
//! Optional storm and shear toggles exaggerate that profile; density variation
//! is applied in the atmosphere lookup (still synthetic, never METAR).

use crate::earth::enu_to_ecef_vec;
use crate::math::{cos, exp, lerp, ln, sin, sqrt, Vec3};
use rand::Rng;

/// Independent sim-weather knobs on top of the layered climatology.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Weather {
    pub storm: bool,
    pub shear: bool,
}

impl Weather {
    /// Multiplier on US76 density. Storm = thicker low-level air; shear-only
    /// days run a slightly thinner mid-atmosphere (stronger tropopause contrast).
    pub fn density_scale(self) -> f64 {
        let mut d = 1.0;
        if self.storm {
            d *= 1.10;
        }
        if self.shear {
            d *= 0.96;
        }
        d
    }
}

#[derive(Clone, Debug)]
pub struct Wind {
    pub scale: f64,
    pub weather: Weather,
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

/// Extra shear-mode layer contrast: (alt_m, speed_mul, dir_offset_deg).
const SHEAR_LAYERS: [(f64, f64, f64); 8] = [
    (0.0, 1.15, 25.0),
    (2_000.0, 0.70, 40.0),
    (6_000.0, 1.55, -55.0),
    (10_000.0, 1.85, -20.0),
    (16_000.0, 1.25, 10.0),
    (24_000.0, 0.80, 30.0),
    (40_000.0, 0.90, 15.0),
    (80_000.0, 1.00, 0.0),
];

impl Wind {
    pub fn new(seed: u64, scale: f64) -> Self {
        Self::new_weather(seed, scale, Weather::default())
    }

    pub fn new_weather(seed: u64, scale: f64, weather: Weather) -> Self {
        Self {
            scale,
            weather,
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
        sqrt(-2.0 * ln(u1)) * cos(2.0 * std::f64::consts::PI * u2)
    }

    pub fn step(&mut self, dt: f64, rng: &mut impl Rng) {
        let _ = rng;
        // τ ≈ 4 s gust memory; storm shortens memory and raises σ.
        let tau = if self.weather.storm { 2.4 } else { 4.0 };
        let mut sigma = 4.0 * self.scale;
        if self.weather.storm {
            sigma *= 2.4;
        }
        if self.weather.shear {
            sigma *= 1.25;
        }
        let a = exp(-dt / tau);
        self.gust_enu = self.gust_enu * a
            + Vec3::new(self.next_gauss(), self.next_gauss(), self.next_gauss() * 0.3)
                * (sigma * sqrt(1.0 - a * a));
    }

    pub fn mean_enu(&self, alt: f64) -> Vec3 {
        let a = alt.max(0.0);
        let mut i = 0;
        while i + 1 < LAYERS.len() && a >= LAYERS[i + 1].0 {
            i += 1;
        }
        let (h0, s0, d0) = self.layer(i);
        let (h1, s1, d1) = if i + 1 < LAYERS.len() {
            self.layer(i + 1)
        } else {
            self.layer(i)
        };
        let t = if (h1 - h0).abs() < 1.0 {
            0.0
        } else {
            ((a - h0) / (h1 - h0)).clamp(0.0, 1.0)
        };
        let mut speed = lerp(s0, s1, t) * self.scale;
        let mut dir = lerp(d0, d1, t);
        if self.weather.storm && a < 4_000.0 {
            let k = 1.0 - a / 4_000.0;
            speed += 16.0 * k * self.scale;
            dir = lerp(dir, 70.0, 0.55 * k);
        }
        let dir = dir * std::f64::consts::PI / 180.0;
        // "from" → toward
        let toward = dir + std::f64::consts::PI;
        Vec3::new(speed * sin(toward), speed * cos(toward), 0.0)
    }

    fn layer(&self, i: usize) -> (f64, f64, f64) {
        let (h, s, d) = LAYERS[i];
        if !self.weather.shear {
            return (h, s, d);
        }
        let (_, sm, d_off) = SHEAR_LAYERS[i];
        (h, s * sm, d + d_off)
    }

    pub fn velocity_ecef(&self, alt: f64, lat: f64, lon: f64) -> Vec3 {
        let enu = self.mean_enu(alt) + self.gust_enu;
        enu_to_ecef_vec(enu, lat, lon)
    }

    pub fn gust_speed(&self) -> f64 {
        self.gust_enu.norm()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storm_is_stronger_near_surface() {
        let fair = Wind::new(1, 1.0);
        let storm = Wind::new_weather(1, 1.0, Weather { storm: true, shear: false });
        assert!(storm.mean_enu(200.0).norm() > fair.mean_enu(200.0).norm() + 8.0);
        assert!((Weather { storm: true, shear: false }.density_scale() - 1.10).abs() < 1e-9);
    }

    #[test]
    fn shear_changes_jet_direction() {
        let fair = Wind::new(1, 1.0);
        let shear = Wind::new_weather(1, 1.0, Weather { storm: false, shear: true });
        let df = fair.mean_enu(10_000.0);
        let ds = shear.mean_enu(10_000.0);
        assert!(ds.norm() > df.norm());
        let align = df.normalized().dot(ds.normalized());
        assert!(align < 0.97, "shear should twist the jet, align={align}");
    }
}
