//! Benchmark CPU vs GPU ensemble throughput (feature `gpu`).
//!
//!   scripts/run-bounded.sh cargo run --release --features gpu --example bench_throughput > bench.csv
//!
//! Times a free-diffusion ensemble (the integrator + RNG, no neighbour search)
//! end to end through each backend and reports walkers per second.

#[cfg(feature = "gpu")]
fn main() {
    use ermak::backend::{CpuBackend, EnsembleBackend, Scenario};
    use ermak::crowding::cubic_lattice;
    use ermak::gpu::GpuBackend;
    use ermak::memory::MemoryBudget;
    use std::time::Instant;

    // A crowded, compute-heavy workload: each step does a WCA force evaluation
    // against every crowder, so the per-walker arithmetic dominates the launch
    // and transfer cost.
    let box_l = 8.0;
    let scenario = Scenario {
        d0: 1.0,
        dt: 0.0002,
        steps: 3_000,
        box_l,
        sigma: 1.0,
        eps: 1.0,
        crowders: cubic_lattice(box_l, 5), // 125 obstacles
    };
    let n = 40_000usize;
    let seed = 1u64;
    let budget = MemoryBudget::new(1usize << 34, "bench");

    let gpu = GpuBackend::new().expect("CUDA device");
    // warm up both kernels (compile/load + first launch).
    let _ = gpu.msd_sum(&scenario, 2_000, seed, &budget).unwrap();
    let _ = gpu.msd_sum_f32(&scenario, 2_000, seed, &budget).unwrap();

    let t = Instant::now();
    let _ = CpuBackend.msd_sum(&scenario, n, seed, &budget).unwrap();
    let cpu_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let _ = gpu.msd_sum(&scenario, n, seed, &budget).unwrap();
    let gpu64_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let _ = gpu.msd_sum_f32(&scenario, n, seed, &budget).unwrap();
    let gpu32_s = t.elapsed().as_secs_f64();

    let wps = |s: f64| n as f64 / s;
    eprintln!(
        "n={n} steps={} crowders={}: CPU {cpu_s:.3}s, GPU-f64 {gpu64_s:.3}s ({:.1}x), GPU-f32 {gpu32_s:.3}s ({:.1}x)",
        scenario.steps,
        scenario.crowders.len(),
        cpu_s / gpu64_s,
        cpu_s / gpu32_s
    );
    println!("backend,walkers_per_sec,seconds");
    println!("CPU,{:.0},{cpu_s:.5}", wps(cpu_s));
    println!("GPU (f64),{:.0},{gpu64_s:.5}", wps(gpu64_s));
    println!("GPU (f32),{:.0},{gpu32_s:.5}", wps(gpu32_s));
}

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("build with --features gpu");
}
