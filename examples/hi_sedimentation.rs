//! Periodic suspension sedimentation coefficient vs theory/MD: the payoff of the
//! periodic Beenakker-Ewald mobility (Milestone B).
//!
//! Apply an equal force to every particle; the mean settling velocity is
//! U_sed = (1/N) sum_i sum_j mu_ij . F. Hindered settling: U_sed/U0 < 1, with the
//! dilute (Batchelor) limit U_sed/U0 = 1 - 6.55 phi for random hard spheres with HI.
//!
//! mu0 = 1/(6a) in the GRPerY units used by the periodic mobility, so U_sed/U0 is
//! that ratio directly. Averaged over random configurations and the 3 force axes.
//!
//! Run: cargo run --release --example hi_sedimentation

use ermak::hydro::ewald::{EwaldParams, periodic_grand_mobility};
use ermak::vec3::Vec3;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn random_suspension(n: usize, l: f64, seed: u64) -> Vec<Vec3> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut pos: Vec<Vec3> = Vec::with_capacity(n);
    let mut tries = 0;
    while pos.len() < n && tries < 1_000_000 {
        tries += 1;
        let p = Vec3::new(
            rng.gen_range(0.0..l),
            rng.gen_range(0.0..l),
            rng.gen_range(0.0..l),
        );
        // minimum-image hard-sphere check (no overlap, gap >= 2a)
        let ok = pos.iter().all(|q| {
            let mut d = p - *q;
            d = Vec3::new(
                d.x - l * (d.x / l).round(),
                d.y - l * (d.y / l).round(),
                d.z - l * (d.z / l).round(),
            );
            d.norm2().sqrt() > 2.0
        });
        if ok {
            pos.push(p);
        }
    }
    pos
}

/// U_sed/U0 for one configuration, averaged over the 3 force directions.
fn u_sed_over_u0(pos: &[Vec3], ep: &EwaldParams) -> f64 {
    let n = pos.len();
    let dim = 3 * n;
    let m = periodic_grand_mobility(pos, ep);
    let mu0 = 1.0 / (6.0 * ep.a); // GRPerY units
    let mut acc = 0.0;
    for axis in 0..3 {
        // F = e_axis on every particle; U_i.axis = sum over all columns in that axis
        let mut usum = 0.0;
        for i in 0..n {
            let row = 3 * i + axis;
            for j in 0..n {
                usum += m[row * dim + (3 * j + axis)];
            }
        }
        acc += usum / n as f64; // mean velocity component along the force
    }
    (acc / 3.0) / mu0
}

fn main() {
    let a = 1.0_f64;
    let n = 20usize;
    println!("# periodic suspension sedimentation, N = {n}, a = {a}");
    println!("# phi    U_sed/U0 (engine)   Batchelor 1-6.55phi   note");
    for &phi in &[0.02_f64, 0.04, 0.06, 0.08, 0.10] {
        let vol = n as f64 * (4.0 / 3.0) * std::f64::consts::PI * a.powi(3) / phi;
        let l = vol.cbrt();
        let ep = EwaldParams {
            box_l: l,
            sigma: l / 4.0,
            r_cut: 1.5 * l,
            k_max: 7,
            a,
        };
        let reps = 6;
        let mut acc = 0.0;
        for s in 0..reps {
            let pos = random_suspension(n, l, 7 + s);
            acc += u_sed_over_u0(&pos, &ep);
        }
        let u = acc / reps as f64;
        let batch = 1.0 - 6.55 * phi;
        println!(
            "{phi:5.2}   {u:16.4}   {batch:19.4}   {}",
            if (u - batch).abs() < 0.06 {
                "consistent"
            } else {
                "see caveat"
            }
        );
    }
    println!();
    println!("# Hindered settling: U_sed/U0 falls below 1 with phi (HI back-flow from");
    println!("# neighbours). Dilute slope ~ Batchelor -6.55 phi; finite-N and the random");
    println!("# (non-equilibrium) configs give scatter. Free-draining ermak would give 1.");
}
