use spacey_sim::sim::Sim;

fn main() {
    for seed in [1u32, 7] {
        let mut sim = Sim::new(seed, true, 1.0);
        let mut steps = 0;
        let mut max_rate = 0.0f64;
        println!("--- seed {seed} ---");
        while !sim.terminated() && steps < 40000 {
            sim.step(sim.adaptive_dt());
            max_rate = max_rate.max(sim.omega_body.norm());
            if steps % 40 == 0 || sim.terminated() {
                println!(
                    "t={:6.1} alt={:7.0} spd={:6.0} rng={:7.0} q={:7.0} aoa={:5.1} tilt={:5.1} thr={:.2} n={} fuel={:5.0} rate={:.2} phase={} term={}",
                    sim.t,
                    sim.last_nav.alt,
                    sim.last_nav.speed,
                    sim.last_nav.range_h,
                    sim.last_aero_q,
                    sim.last_aoa * 180.0 / std::f64::consts::PI,
                    sim.last_nav.tilt * 180.0 / std::f64::consts::PI,
                    sim.last_throttle,
                    sim.last_n_engines,
                    sim.fuel,
                    sim.omega_body.norm(),
                    sim.phase.as_str(),
                    sim.term.as_str()
                );
            }
            steps += 1;
        }
        println!(
            "END steps={steps} max_rate={max_rate:.2} dest='{}'\n",
            sim.destroy_reason.as_str()
        );
    }
}
