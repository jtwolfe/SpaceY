//! Quick LEO landing-theater summary across seeds.
//! `cargo run -p spacey_sim --example leo_land --release`

use spacey_sim::guidance::{Phase, TermReason};
use spacey_sim::scenario::Scenario;
use spacey_sim::sim::Sim;
use spacey_sim::wind::Weather;

fn main() {
    let seeds = [3u32, 20, 54, 88, 7, 11, 33, 41];
    println!(
        "{:>5} {:>8} {:>7} {:>7} {:>7} {:>6} {:>7} {:>8} {:<10} {}",
        "seed", "t", "alt", "spd", "gc", "fuel", "minGC", "minGCalt", "term", "dest"
    );
    for seed in seeds {
        let mut sim = Sim::new_with(seed, true, 0.0, Scenario::LeoDeorbit, Weather::default());
        let mut min_gc = f64::INFINITY;
        let mut min_gc_alt = 0.0;
        let mut min_gc_spd = 0.0;
        let mut min_gc_fuel = 0.0;
        let mut guard = 0;
        while !sim.terminated() && guard < 140_000 {
            sim.step(sim.adaptive_dt());
            if sim.last_nav.range_gc < min_gc {
                min_gc = sim.last_nav.range_gc;
                min_gc_alt = sim.last_nav.alt;
                min_gc_spd = sim.last_nav.speed;
                min_gc_fuel = sim.fuel;
            }
            guard += 1;
        }
        println!(
            "{:>5} {:>8.1} {:>7.2} {:>7.0} {:>7.1} {:>6.2} {:>7.1} {:>8.1} {:<10} {}  land={:?} intact={}",
            seed,
            sim.t,
            sim.last_nav.alt / 1000.0,
            sim.last_nav.speed,
            sim.last_nav.range_gc / 1000.0,
            sim.fuel / 1000.0,
            min_gc / 1000.0,
            min_gc_alt / 1000.0,
            sim.term.as_str(),
            sim.destroy_reason.as_str(),
            sim.phase,
            sim.intact
        );
        println!(
            "      closest: alt={:.1} km spd={:.0} fuel={:.1} t",
            min_gc_alt / 1000.0,
            min_gc_spd,
            min_gc_fuel / 1000.0
        );
        let _ = Phase::Landing;
        let _ = TermReason::Success;
    }
}
