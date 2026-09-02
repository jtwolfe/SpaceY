//! Time-series of one LEO nominal through entry. Release-only diagnostic.
//! `cargo run -p spacey_sim --example leo_trace --release`

use spacey_sim::scenario::Scenario;
use spacey_sim::sim::Sim;
use spacey_sim::wind::Weather;

fn main() {
    let mut sim = Sim::new_with(3, true, 0.0, Scenario::LeoDeorbit, Weather::default());
    let mut last_print = -100.0;
    let mut min_gc = sim.last_nav.range_gc;
    let mut min_gc_alt = sim.last_nav.alt;
    let mut min_gc_spd = sim.last_nav.speed;
    let mut min_gc_t = 0.0;
    let mut min_gc_fuel = sim.fuel;
    println!(
        "{:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>7} {:>6} {:>4} {:<8}",
        "t", "alt_km", "spd", "gc_km", "east_km", "north_km", "Q_kPa", "fuel_t", "thr", "n", "phase"
    );
    while !sim.terminated() {
        sim.step(sim.adaptive_dt());
        if sim.last_nav.range_gc < min_gc {
            min_gc = sim.last_nav.range_gc;
            min_gc_alt = sim.last_nav.alt;
            min_gc_spd = sim.last_nav.speed;
            min_gc_t = sim.t;
            min_gc_fuel = sim.fuel;
        }
        let in_entry = sim.t > 2480.0 || sim.last_nav.range_gc < 1_200_000.0;
        if in_entry && (sim.t - last_print > 8.0 || sim.terminated()) {
            last_print = sim.t;
            println!(
                "{:>8.1} {:>8.2} {:>8.0} {:>8.1} {:>8.1} {:>8.1} {:>7.1} {:>6.2} {:>6.2} {:>4} {:<8}",
                sim.t,
                sim.last_nav.alt / 1000.0,
                sim.last_nav.speed,
                sim.last_nav.range_gc / 1000.0,
                sim.last_nav.pos_enu.x / 1000.0,
                sim.last_nav.pos_enu.y / 1000.0,
                sim.last_aero_q / 1000.0,
                sim.fuel / 1000.0,
                sim.last_throttle,
                sim.last_n_engines,
                sim.phase.as_str()
            );
        }
    }
    println!(
        "\nminGC={:.1} km @ t={:.1} alt={:.1} km spd={:.0} fuel={:.1} t",
        min_gc / 1000.0,
        min_gc_t,
        min_gc_alt / 1000.0,
        min_gc_spd,
        min_gc_fuel / 1000.0
    );
    println!(
        "term={} dest='{}' phase={} alt={:.2} km spd={:.0} range_h={:.1} km fuel={:.2} t intact={}",
        sim.term.as_str(),
        sim.destroy_reason.as_str(),
        sim.phase.as_str(),
        sim.last_nav.alt / 1000.0,
        sim.last_nav.speed,
        sim.last_nav.range_h / 1000.0,
        sim.fuel / 1000.0,
        sim.intact
    );
}
