//! HI vs free-draining: short-time self-diffusion of a tracer among FIXED
//! obstacles, the direct new-engine-vs-previous-ermak crowding comparison.
//!
//! With obstacles held fixed, the tracer's rigorous effective mobility is the
//! Schur complement mu_eff = ((M^-1)_tt)^-1, where M is the grand RPY mobility
//! and tt is the tracer's 3x3 block. Short-time self-diffusion D_s = kT tr(mu_eff)/3.
//!
//!   - HI ON  (new engine): obstacles hydrodynamically hinder the tracer,
//!     D_s/D0 < 1, decreasing with obstacle volume fraction phi.
//!   - HI OFF (previous free-draining ermak): the tracer's mobility is mu0
//!     regardless of obstacles, so D_s/D0 = 1 (steric slowdown only shows up in
//!     the LONG-time, dynamic MSD of crowding.rs, not in short-time self-diffusion).
//!
//! The gap between the two columns is the hydrodynamic hindrance that the
//! previous ermak could not capture.
//!
//! Run: cargo run --release --example hi_crowding

use ermak::hydro::HydroSystem;
use ermak::hydro::mobility::{cholesky, grand_mobility};
use ermak::vec3::Vec3;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::Rng;

/// Solve L L^T x = b (lower-Cholesky factor L, dim x dim, row-major).
fn chol_solve(l: &[f64], dim: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0f64; dim];
    for i in 0..dim {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * dim + k] * y[k];
        }
        y[i] = s / l[i * dim + i];
    }
    let mut x = vec![0.0f64; dim];
    for i in (0..dim).rev() {
        let mut s = y[i];
        for k in (i + 1)..dim {
            s -= l[k * dim + i] * x[k];
        }
        x[i] = s / l[i * dim + i];
    }
    x
}

/// Inverse of a symmetric 3x3 (row-major 9).
fn inv3(m: [f64; 9]) -> [f64; 9] {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    let id = 1.0 / det;
    [
        (m[4] * m[8] - m[5] * m[7]) * id,
        (m[2] * m[7] - m[1] * m[8]) * id,
        (m[1] * m[5] - m[2] * m[4]) * id,
        (m[5] * m[6] - m[3] * m[8]) * id,
        (m[0] * m[8] - m[2] * m[6]) * id,
        (m[2] * m[3] - m[0] * m[5]) * id,
        (m[3] * m[7] - m[4] * m[6]) * id,
        (m[1] * m[6] - m[0] * m[7]) * id,
        (m[0] * m[4] - m[1] * m[3]) * id,
    ]
}

/// Short-time tracer self-diffusion D_s/D0 among fixed obstacles (HI on).
/// Tracer is particle 0; obstacles fill a cubic box at volume fraction phi,
/// placed with a minimum surface gap so they do not overlap the tracer.
fn ds_over_d0(n_obst: usize, phi: f64, seed: u64) -> f64 {
    let a = 1.0;
    let eta = 1.0;
    let mu0 = 1.0 / (6.0 * std::f64::consts::PI * eta * a);
    let vol = (n_obst as f64 + 1.0) * (4.0 / 3.0) * std::f64::consts::PI * a.powi(3) / phi;
    let l = vol.cbrt();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut pos = vec![Vec3::new(l * 0.5, l * 0.5, l * 0.5)]; // tracer at centre
    let mut tries = 0;
    while pos.len() < n_obst + 1 && tries < 200_000 {
        tries += 1;
        let p = Vec3::new(rng.gen_range(0.0..l), rng.gen_range(0.0..l), rng.gen_range(0.0..l));
        // keep a small surface gap (2.1a centre-centre) so RPY stays near-field-clean
        if pos.iter().all(|q| (p - *q).norm2().sqrt() > 2.1 * a) {
            pos.push(p);
        }
    }
    let n = pos.len();
    let dim = 3 * n;
    let sys = HydroSystem { pos, radius: vec![a; n], charge: vec![0.0; n], eta, kt: 1.0, box_l: None };
    let m = grand_mobility(&sys);
    let l_chol = cholesky(&m, dim).expect("SPD");
    // (M^-1)_tt: solve M x = e_d for d in 0..3, take the top 3 rows
    let mut rtt = [0.0f64; 9];
    for d in 0..3 {
        let mut b = vec![0.0f64; dim];
        b[d] = 1.0;
        let x = chol_solve(&l_chol, dim, &b);
        for r in 0..3 {
            rtt[3 * r + d] = x[r];
        }
    }
    let mu_eff = inv3(rtt);
    let tr = mu_eff[0] + mu_eff[4] + mu_eff[8];
    (tr / 3.0) / mu0
}

fn main() {
    println!("# short-time tracer self-diffusion among FIXED obstacles");
    println!("# phi   n_obst   D_s/D0 (HI, new)   D_s/D0 (free-draining, old ermak)   hindrance");
    let n_obst = 60;
    for &phi in &[0.05_f64, 0.10, 0.15, 0.20, 0.25, 0.30] {
        let reps = 8;
        let mut acc = 0.0;
        for s in 0..reps {
            acc += ds_over_d0(n_obst, phi, 100 + s);
        }
        let ds_hi = acc / reps as f64;
        let hindrance = 1.0 - ds_hi;
        println!(
            "{phi:4.2}   {n_obst:6}   {ds_hi:15.4}   {:33.1}   {:.1}% slower",
            1.0,
            hindrance * 100.0
        );
    }
    println!();
    println!("# free-draining (old ermak) short-time self-diffusion is D0 at all phi;");
    println!("# the HI column is the hydrodynamic crowding hindrance it cannot capture.");
    println!("# (steric/excluded-volume slowdown is a separate LONG-time effect, crowding.rs.)");
}
