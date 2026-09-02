//! Print seed-88 (and friends) landing residuals for debug/release comparison.
//! `cargo run -p spacey_sim --example leo_cmp`
//! `cargo run -p spacey_sim --example leo_cmp --release`

use spacey_sim::scenario::Scenario;
use spacey_sim::sim::Sim;
use spacey_sim::wind::Weather;

fn main() {
    let seeds: Vec<u32> = std::env::args()
        .skip(1)
        .filter_map(|s| s.parse().ok())
        .collect();
    let seeds = if seeds.is_empty() {
        vec![88u32, 3, 20, 54]
    } else {
        seeds
    };
    for seed in seeds {
        let mut sim = Sim::new_with(seed, true, 0.0, Scenario::LeoDeorbit, Weather::default());
        let until = std::env::var("UNTIL")
            .ok()
            .and_then(|s| s.parse::<f64>().ok());
        let mut guard = 0u32;
        let mut min_gc = f64::INFINITY;
        let mut _at_min = (0.0, 0.0, 0.0, 0.0);
        while !sim.terminated() && guard < 140_000 {
            if until.map(|u| sim.t >= u).unwrap_or(false) {
                break;
            }
            sim.step(sim.adaptive_dt());
            if sim.last_nav.range_gc < min_gc {
                min_gc = sim.last_nav.range_gc;
                _at_min = (
                    sim.t,
                    sim.last_nav.alt,
                    sim.last_nav.speed,
                    sim.fuel,
                );
            }
            guard += 1;
        }
        let snap = sim.snapshot();
        println!(
            "seed={seed} steps={guard} t={:.4} term={} dest='{}' phase={} alt={:.4} lat={:.6} lon={:.6} spd={:.4} vin={:.4} east={:.2} north={:.2} rh={:.2} gc={:.2} fuel={:.2} peri={:.1} r_eci=[{:.4},{:.4},{:.4}]",
            sim.t,
            sim.term.as_str(),
            sim.destroy_reason.as_str(),
            sim.phase.as_str(),
            snap.alt,
            snap.lat,
            snap.lon,
            sim.last_nav.speed,
            snap.speed_inertial,
            sim.last_nav.pos_enu.x,
            sim.last_nav.pos_enu.y,
            sim.last_nav.range_h,
            snap.range_gc,
            sim.fuel,
            snap.periapsis_alt,
            snap.r_eci[0],
            snap.r_eci[1],
            snap.r_eci[2],
        );
    }
}
