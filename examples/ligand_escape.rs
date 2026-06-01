//! Ligand escape from a buried pocket: residence time vs the bottleneck barrier,
//! and tauRAMD egress times that rank those residence times.
//!
//! Lifted from Nunes-Alves's dissociation-kinetics work (ligand escape from
//! T4 lysozyme; inhibitor dissociation through NiFe-hydrogenase bottlenecks):
//! a coarse-grained ligand sits in a buried pocket and must cross a bottleneck
//! barrier to dissociate. Raising the barrier is a proxy for a congeneric
//! series of slower-dissociating ligands. tauRAMD adds a random-acceleration
//! force that drives escape fast; its egress times are not the true residence
//! times but rank them, the property that makes tauRAMD a practical predictor
//! of relative k_off.
//!
//!   cargo run --release --example ligand_escape > escape.csv
//!   python scripts/plot_escape.py escape.csv
//!
//! Reduced units (kB T = 1, D_0 = 1).

use ermak::kinetics::{mean_residence_time, tauramd_egress_time};

fn main() {
    let (r_b, d0, dt) = (2.0, 1.0, 0.001);
    let max_steps = 300_000usize;
    let reps = 600usize;
    let seed = 1u64;
    let (accel, reorient) = (4.0, 100usize);

    println!("barrier,residence_time,tauramd_time");
    for &barrier in &[1.0, 2.0, 3.0, 4.0, 5.0] {
        let residence = mean_residence_time(barrier, r_b, d0, dt, max_steps, reps, seed);
        let tau = tauramd_egress_time(
            barrier,
            r_b,
            d0,
            dt,
            accel,
            reorient,
            max_steps,
            reps,
            seed + 1,
        );
        println!("{barrier:.1},{residence:.4},{tau:.4}");
    }
}
