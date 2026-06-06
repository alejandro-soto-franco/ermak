//! Scaling harness for the positively-split sinc^2 / Hasimoto PSE apply: full-grid
//! vs truncated-support, at fixed number density so the grid grows as
//! `ng ~ N^{1/3}` (hence `ng^3 ~ N`). The full-grid spread/gather is `O(N ng^3) ~
//! O(N^2)`; the truncated window is `O(N P^3) ~ O(N)`. This prints wall-clock per
//! apply for each so the crossover and the linear-vs-quadratic trend are visible.
//!
//! Run: `cargo run --release --features gpu --example pse_sinc_scaling`
//! (needs a CUDA GPU; findings are recorded in the ermak-planning repo).

#[cfg(feature = "gpu")]
fn main() {
    use ermak::hydro::ewald::EwaldParams;
    use ermak::hydro::gpu_pse_wave::GpuPseWave;
    use ermak::hydro::pse_wave::WaveParams;
    use ermak::vec3::Vec3;
    use std::time::Instant;

    // Fixed spacing between particles on a cubic lattice (well separated), and a
    // fixed grid spacing h, so both the box and ng scale as N^{1/3}.
    let spacing = 2.5_f64;
    let h_target = 0.6_f64;
    let support = 5usize;
    let reps = 20usize;

    let dev = GpuPseWave::new().expect("cuda device");
    println!(
        "{:>6} {:>5} {:>8} {:>12} {:>12} {:>9}",
        "N", "ng", "box_l", "full_ms", "trunc_ms", "speedup"
    );

    for &m in &[3usize, 4, 5, 6, 7, 8] {
        let n = m * m * m;
        let box_l = m as f64 * spacing;
        // even ng nearest to box_l / h_target
        let mut ng = (box_l / h_target).round() as usize;
        if ng % 2 == 1 {
            ng += 1;
        }
        let h = box_l / ng as f64;

        // particles on the m^3 lattice, a small deterministic jitter to avoid exact
        // symmetry; forces are a fixed pseudo-random pattern.
        let mut pos = Vec::with_capacity(n);
        let mut forces = Vec::with_capacity(n);
        let mut seed = 1u64;
        let mut next = || {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 - 1.0 // in (-1, 1)
        };
        for i in 0..m {
            for j in 0..m {
                for k in 0..m {
                    let x = i as f64 * spacing + 0.1 * next();
                    let y = j as f64 * spacing + 0.1 * next();
                    let z = k as f64 * spacing + 0.1 * next();
                    pos.push(Vec3::new(x, y, z));
                    forces.push(Vec3::new(next(), next(), next()));
                }
            }
        }

        let ep = EwaldParams {
            box_l,
            sigma: 2.5,
            r_cut: 13.0,
            k_max: 12,
            a: 1.0,
        };
        let wp_trunc = WaveParams::truncated(box_l, ep.sigma, ep.a, ng, h, support);

        // warm up (plan + buffer arena build) and time the median-ish mean.
        let _ = dev.full_apply_sinc(&pos, &forces, &ep, ng).expect("full");
        let t0 = Instant::now();
        for _ in 0..reps {
            let _ = dev.full_apply_sinc(&pos, &forces, &ep, ng).expect("full");
        }
        let full_ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;

        let _ = dev
            .full_apply_sinc_wp(&pos, &forces, &ep, &wp_trunc)
            .expect("trunc");
        let t1 = Instant::now();
        for _ in 0..reps {
            let _ = dev
                .full_apply_sinc_wp(&pos, &forces, &ep, &wp_trunc)
                .expect("trunc");
        }
        let trunc_ms = t1.elapsed().as_secs_f64() * 1e3 / reps as f64;

        println!(
            "{n:>6} {ng:>5} {box_l:>8.1} {full_ms:>12.3} {trunc_ms:>12.3} {:>8.2}x",
            full_ms / trunc_ms
        );
    }
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("build with --features gpu to run the sinc^2 PSE scaling harness");
}
