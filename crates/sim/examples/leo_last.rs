//! Last-kilometre LEO trace.
//! `cargo run -p spacey_sim --example leo_last --release -- 88`

use spacey_sim::scenario::Scenario;
use spacey_sim::sim::Sim;
use spacey_sim::wind::Weather;

fn main() {
    let seed: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(88);
    let mut sim = Sim::new_with(seed, true, 0.0, Scenario::LeoDeorbit, Weather::default());
    let mut last = -100.0;
    while !sim.terminated() {
        sim.step(sim.adaptive_dt());
        let n = &sim.last_nav;
        let close = n.range_gc < 20_000.0 || n.alt < 8_000.0 || sim.phase.as_str() == "LANDING";
        let interval = if n.engine_alt < 50.0 { 0.04 } else { 1.2 };
        if close && (sim.t - last > interval || sim.terminated()) {
            last = sim.t;
            println!(
                "t={:.1} phase={} alt={:.0} ealt={:.0} spd={:.1} vh={:.1} vz={:.1} east={:.0} north={:.0} rh={:.0} gc={:.0} fuel={:.0} thr={:.2} n={} q={:.0} tilt={:.1}",
                sim.t,
                sim.phase.as_str(),
                n.alt,
                n.engine_alt,
                n.speed,
                (n.v_enu.x * n.v_enu.x + n.v_enu.y * n.v_enu.y).sqrt(),
                n.v_enu.z,
                n.pos_enu.x,
                n.pos_enu.y,
                n.range_h,
                n.range_gc,
                sim.fuel,
                sim.last_throttle,
                sim.last_n_engines,
                sim.last_aero_q,
                n.tilt * 180.0 / std::f64::consts::PI,
            );
        }
    }
    println!(
        "TERM {} dest='{}' intact={}",
        sim.term.as_str(),
        sim.destroy_reason.as_str(),
        sim.intact
    );
}
