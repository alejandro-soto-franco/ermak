//! Stress test for the dense (oracle) RPY engine: cost scaling and numerical
//! stability. The dense path is O((3N)^3) per step (Cholesky dominates), so this
//! maps the practical N ceiling and confirms the mobility stays SPD at dense
//! packing and near-contact (the overlap branch of the RPY kernel).
//!
//! Run: cargo run --release --example hi_stress

use ermak::hydro::HydroSystem;
use ermak::hydro::mobility::{cholesky, grand_mobility};
use ermak::vec3::Vec3;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::Rng;
use std::time::Instant;

fn random_box(n: usize, box_l: f64, seed: u64) -> HydroSystem {
    let mut rng = StdRng::seed_from_u64(seed);
    let pos = (0..n)
        .map(|_| Vec3::new(rng.gen_range(0.0..box_l), rng.gen_range(0.0..box_l), rng.gen_range(0.0..box_l)))
        .collect();
    HydroSystem { pos, radius: vec![1.0; n], charge: vec![0.0; n], eta: 1.0, kt: 1.0, box_l: None }
}

fn main() {
    println!("# RPY dense-engine stress test (open boundary, a=1)");
    println!("# N    dim   build_ms  chol_ms   total_ms   ms/step   SPD");
    for &n in &[10usize, 25, 50, 100, 200, 400] {
        // box sized so volume fraction phi ~ 0.1 (N * (4/3 pi a^3) / L^3)
        let vol = n as f64 * (4.0 / 3.0) * std::f64::consts::PI / 0.1;
        let l = vol.cbrt();
        let sys = random_box(n, l, 42);
        let dim = 3 * n;

        let t0 = Instant::now();
        let m = grand_mobility(&sys);
        let build = t0.elapsed().as_secs_f64() * 1e3;

        let t1 = Instant::now();
        let spd = cholesky(&m, dim).is_ok();
        let chol = t1.elapsed().as_secs_f64() * 1e3;

        println!(
            "{n:5} {dim:5} {build:9.2} {chol:8.2} {:9.2} {:9.2}   {}",
            build + chol,
            build + chol,
            if spd { "yes" } else { "NO" }
        );
    }

    // O(N^3) check: doubling N should ~8x the Cholesky time (last two rows above).
    println!();
    println!("# stability at high packing + near-contact (overlap branch):");
    for &phi in &[0.2_f64, 0.3, 0.4] {
        let n = 80usize;
        let vol = n as f64 * (4.0 / 3.0) * std::f64::consts::PI / phi;
        let l = vol.cbrt();
        let mut worst_min_gap = f64::INFINITY;
        let mut all_spd = true;
        for seed in 0..10u64 {
            let sys = random_box(n, l, seed);
            // record the closest pair (probes the RPY overlap branch)
            for i in 0..n {
                for j in (i + 1)..n {
                    let d = (sys.pos[i] - sys.pos[j]).norm2().sqrt();
                    worst_min_gap = worst_min_gap.min(d);
                }
            }
            let m = grand_mobility(&sys);
            if cholesky(&m, 3 * n).is_err() {
                all_spd = false;
            }
        }
        println!(
            "  phi={phi:.1}  N={n}  closest pair seen = {worst_min_gap:.3} (2a=2.0)  SPD over 10 configs: {}",
            if all_spd { "yes" } else { "NO" }
        );
    }
}
