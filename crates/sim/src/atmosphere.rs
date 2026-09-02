//! US Standard Atmosphere 1976, 0–86 km, plus a simple exponential tail.
//!
//! Coefficients from the official U.S. Standard Atmosphere, 1976 (NASA-TM-X-74335 /
//! NOAA-S/T 76-1562). Geometric altitude is converted to geopotential height
//! with the document's r0 = 6 356 766 m.

use crate::constants::{GAMMA_AIR, G0, R_SPECIFIC_AIR};

const R0: f64 = 6_356_766.0;
const G_R: f64 = G0 / R_SPECIFIC_AIR;

#[derive(Clone, Copy, Debug)]
pub struct Air {
    pub temperature_k: f64,
    pub pressure_pa: f64,
    pub density: f64,
    pub speed_of_sound: f64,
}

/// Layer bases: geopotential height (m), lapse rate (K/m), T_b (K), p_b (Pa).
const LAYERS: [(f64, f64, f64, f64); 7] = [
    (0.0, -0.0065, 288.15, 101_325.0),
    (11_000.0, 0.0, 216.65, 22_632.1),
    (20_000.0, 0.0010, 216.65, 5_474.89),
    (32_000.0, 0.0028, 228.65, 868.019),
    (47_000.0, 0.0, 270.65, 110.906),
    (51_000.0, -0.0028, 270.65, 66.9389),
    (71_000.0, -0.0020, 214.65, 3.95642),
];
const H_TOP: f64 = 84_852.0;
const T_TOP: f64 = 186.87;
const P_TOP: f64 = 0.3734;

fn geopotential(h_geom: f64) -> f64 {
    R0 * h_geom / (R0 + h_geom)
}

/// US76 lookup with a uniform density/pressure scale (synthetic weather day).
pub fn lookup_scaled(altitude_m: f64, density_scale: f64) -> Air {
    let mut air = lookup(altitude_m);
    let s = density_scale.clamp(0.2, 2.5);
    air.density *= s;
    air.pressure_pa *= s;
    air
}

pub fn lookup(altitude_m: f64) -> Air {
    let h = altitude_m.max(-200.0);
    if h < 0.0 {
        // Slightly below pad: use sea-level and let ground contact handle it.
        return from_tp(288.15, 101_325.0);
    }
    let h_geop = geopotential(h);
    if h_geop > H_TOP || h > 86_000.0 {
        return thermosphere(h);
    }
    let mut idx = 0;
    for i in 0..LAYERS.len() {
        if h_geop >= LAYERS[i].0 {
            idx = i;
        }
    }
    let (h_b, lapse, t_b, p_b) = LAYERS[idx];
    let dh = h_geop - h_b;
    let t = t_b + lapse * dh;
    let p = if lapse.abs() < 1e-9 {
        p_b * (-G_R * dh / t_b).exp()
    } else {
        p_b * (t / t_b).powf(-G0 / (lapse * R_SPECIFIC_AIR))
    };
    from_tp(t, p)
}

fn thermosphere(h_geom: f64) -> Air {
    // Crude exponential extension above 86 km. Density is already ~1e-6 of
    // sea level; only needed so exoatmospheric coast is not a hard vacuum cut.
    let h0 = 86_000.0;
    let air86 = from_tp(T_TOP, P_TOP);
    let scale = 6_000.0;
    let f = (-(h_geom - h0).max(0.0) / scale).exp();
    let t = (T_TOP + 0.002 * (h_geom - h0)).min(1000.0);
    from_tp(t, (air86.pressure_pa * f).max(1e-8))
}

fn from_tp(t: f64, p: f64) -> Air {
    let density = p / (R_SPECIFIC_AIR * t);
    let speed_of_sound = (GAMMA_AIR * R_SPECIFIC_AIR * t).sqrt();
    Air {
        temperature_k: t,
        pressure_pa: p,
        density,
        speed_of_sound,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sea_level() {
        let a = lookup(0.0);
        assert!((a.pressure_pa - 101_325.0).abs() < 2.0);
        assert!((a.density - 1.225).abs() < 0.01);
        assert!((a.temperature_k - 288.15).abs() < 0.2);
    }

    #[test]
    fn tropopause() {
        let a = lookup(11_000.0);
        assert!((a.temperature_k - 216.65).abs() < 1.0);
        assert!(a.pressure_pa > 20_000.0 && a.pressure_pa < 24_000.0);
    }

    #[test]
    fn stratosphere_and_vacuum() {
        let mid = lookup(40_000.0);
        assert!(mid.density > 0.003 && mid.density < 0.005);
        let vac = lookup(120_000.0);
        assert!(vac.density < 1e-7);
        let scaled = lookup_scaled(0.0, 1.10);
        assert!((scaled.density - lookup(0.0).density * 1.10).abs() < 1e-6);
    }
}
