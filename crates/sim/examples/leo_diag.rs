//! Measure how the current LEO nominal dies (destruction on, zero residual).
//!
//! Run: `cargo run -p spacey_sim --example leo_diag --release`

use spacey_sim::guidance::TermReason;
use spacey_sim::scenario::Scenario;
use spacey_sim::sim::Sim;
use spacey_sim::wind::Weather;

fn main() {
    let seeds: Vec<u32> = (0..12).map(|i| 3 + i * 17).collect();
    println!("=== LEO nominal, destruction ON, zero residual ===\n");
    println!(
        "{:>6} {:>8} {:>8} {:>8} {:>8} {:>7} {:>6} {:>6} {:>6} {:>8} {:<10} {:<22} {}",
        "seed", "t", "alt_km", "spd", "vin", "Q_kPa", "G", "AoA", "fuel", "range_km", "phase",
        "term", "destroy"
    );

    let mut n_land = 0u32;
    let mut n_entry = 0u32;
    for seed in seeds {
        let mut sim = Sim::new_with(seed, true, 0.0, Scenario::LeoDeorbit, Weather::default());
        let mut steps = 0u32;
        let mut last_phase = sim.phase;
        let mut peak_q = 0.0f64;
        let mut peak_g = 0.0f64;
        let mut peak_aoa_q = 0.0f64;
        let mut min_alt = sim.last_nav.alt;
        let mut min_gc = sim.last_nav.range_gc;
        let mut seen_entry = false;
        let mut seen_glide = false;
        let mut seen_land = false;
        while !sim.terminated() && steps < 120_000 {
            sim.step(sim.adaptive_dt());
            steps += 1;
            peak_q = peak_q.max(sim.last_aero_q);
            peak_g = peak_g.max(sim.last_accel_g);
            if sim.last_aero_q > 200.0 {
                peak_aoa_q = peak_aoa_q.max(sim.last_aoa.abs());
            }
            min_alt = min_alt.min(sim.last_nav.alt);
            min_gc = min_gc.min(sim.last_nav.range_gc);
            if sim.phase != last_phase {
                println!(
                    "  phase {} → {}  t={:.0} alt={:.1}km spd={:.0} peri={:.0}km fuel={:.0} q={:.0} gc={:.0}km",
                    last_phase.as_str(),
                    sim.phase.as_str(),
                    sim.t,
                    sim.last_nav.alt / 1000.0,
                    sim.last_nav.speed,
                    sim.last_nav.periapsis_alt / 1000.0,
                    sim.fuel,
                    sim.last_aero_q,
                    sim.last_nav.range_gc / 1000.0
                );
                last_phase = sim.phase;
            }
            match sim.phase {
                spacey_sim::guidance::Phase::Entry => seen_entry = true,
                spacey_sim::guidance::Phase::Glide => seen_glide = true,
                spacey_sim::guidance::Phase::Landing => seen_land = true,
                _ => {}
            }
        }
        if seen_entry {
            n_entry += 1;
        }
        if seen_land {
            n_land += 1;
        }
        let term = if sim.term == TermReason::None {
            "TIMEOUT_CAP"
        } else {
            sim.term.as_str()
        };
        println!(
            "{:>6} {:>8.1} {:>8.2} {:>8.0} {:>8.0} {:>7.1} {:>6.1} {:>6.1} {:>6.0} {:>8.1} {:<10} {:<22} {}  peaks q={:.0} G={:.1} aoa@q={:.1}° rate={:.2} minAlt={:.1}km minGC={:.0}km entry={} glide={} land={} steps={}",
            seed,
            sim.t,
            sim.last_nav.alt / 1000.0,
            sim.last_nav.speed,
            sim.v_eci.norm(),
            sim.last_aero_q / 1000.0,
            sim.last_accel_g,
            sim.last_aoa * 180.0 / std::f64::consts::PI,
            sim.fuel,
            sim.last_nav.range_h / 1000.0,
            sim.phase.as_str(),
            term,
            sim.destroy_reason.as_str(),
            peak_q,
            peak_g,
            peak_aoa_q * 180.0 / std::f64::consts::PI,
            sim.omega_body.norm(),
            min_alt / 1000.0,
            min_gc / 1000.0,
            seen_entry,
            seen_glide,
            seen_land,
            steps
        );
    }
    println!("\nreached entry: {n_entry}/12   reached landing latch: {n_land}/12");
}
