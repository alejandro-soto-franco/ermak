//! Tracer diffusion in a crowded environment: a Brownian tracer among fixed
//! crowder spheres in a periodic box. The crowders' excluded volume (WCA)
//! obstructs the tracer, reducing its effective diffusion coefficient, the
//! qualitative result of Dey et al. 2022 on crowder-slowed small-molecule
//! diffusion.
//!
//! v1 holds the crowders fixed (a quenched obstacle matrix); mobile crowders
//! are a documented extension.

use crate::potential::wca_pair_force;
use crate::rng::brownian_displacement;
use crate::vec3::Vec3;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

/// Crowder centres on a simple-cubic lattice of `n` per side in a box of side
/// `box_l`, offset to the cell centres so the lattice corners are interstitial
/// voids (a safe tracer start). Volume fraction is `n^3 (pi/6) sigma^3 / L^3`.
#[must_use]
pub fn cubic_lattice(box_l: f64, n: usize) -> Vec<Vec3> {
    let a = box_l / n as f64;
    let mut out = Vec::with_capacity(n * n * n);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                out.push(Vec3::new(
                    (i as f64 + 0.5) * a,
                    (j as f64 + 0.5) * a,
                    (k as f64 + 0.5) * a,
                ));
            }
        }
    }
    out
}

/// Crowder volume fraction `phi = N (pi/6) sigma^3 / L^3` for `n_crowders`
/// spheres of diameter `sigma` in a cubic box of side `box_l`.
#[must_use]
pub fn volume_fraction(n_crowders: usize, sigma: f64, box_l: f64) -> f64 {
    let sphere = std::f64::consts::PI / 6.0 * sigma.powi(3);
    n_crowders as f64 * sphere / box_l.powi(3)
}

/// Minimum-image displacement of `d` under a cubic box of side `l`.
fn min_image(d: Vec3, l: f64) -> Vec3 {
    Vec3::new(
        d.x - l * (d.x / l).round(),
        d.y - l * (d.y / l).round(),
        d.z - l * (d.z / l).round(),
    )
}

/// Effective diffusion coefficient of a tracer (bare coefficient `d0`) among
/// fixed `crowders` in a periodic box of side `box_l`, from the ensemble MSD.
/// Reduced units (`kB T = 1`, so mobility = `d0`).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn crowded_diffusion_deff(
    d0: f64,
    dt: f64,
    steps: usize,
    replicas: usize,
    seed: u64,
    box_l: f64,
    crowders: &[Vec3],
    sigma: f64,
    eps: f64,
) -> f64 {
    use crate::integrator::em_step;

    let total_msd: f64 = (0..replicas)
        .into_par_iter()
        .map(|replica| {
            let mut rng =
                StdRng::seed_from_u64(seed ^ (replica as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            // Start at a lattice corner (interstitial void), unwrapped position.
            let mut r = Vec3::ZERO;
            for _ in 0..steps {
                // Excluded-volume force from every crowder under the minimum image.
                let mut force = Vec3::ZERO;
                for &c in crowders {
                    let d = min_image(r - c, box_l);
                    force += wca_pair_force(d, sigma, eps);
                }
                let noise = brownian_displacement(d0, dt, &mut rng);
                // Reduced units: mobility = d0 / (kB T) = d0.
                r = em_step(r, force, d0, dt, noise);
            }
            r.norm2()
        })
        .sum();
    let msd = total_msd / replicas as f64;
    let t = steps as f64 * dt;
    msd / (6.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn volume_fraction_of_a_lattice() {
        // 216 unit-diameter spheres in an 8^3 box: phi = 216 (pi/6) / 512.
        let phi = volume_fraction(216, 1.0, 8.0);
        let expected = 216.0 * std::f64::consts::PI / 6.0 / 512.0;
        assert!(
            (phi - expected).abs() < 1e-12,
            "phi={phi}, expected={expected}"
        );
    }

    #[test]
    fn crowders_slow_the_tracer() {
        let d0 = 1.0;
        // Hard-core WCA needs a small timestep for stability; t = steps*dt = 2.
        let (dt, steps, replicas, seed) = (0.0002, 10_000usize, 400usize, 7u64);
        let (box_l, sigma, eps) = (8.0, 1.0, 1.0);

        let free = crowded_diffusion_deff(d0, dt, steps, replicas, seed, box_l, &[], sigma, eps);
        let crowders = cubic_lattice(box_l, 6); // 216 obstacles, phi ~ 0.22
        let crowded =
            crowded_diffusion_deff(d0, dt, steps, replicas, seed, box_l, &crowders, sigma, eps);

        assert!(free > 0.9 * d0, "empty box should recover d0: free={free}");
        assert!(
            crowded < 0.85 * free,
            "crowding should slow diffusion: free={free}, crowded={crowded}"
        );
    }
}
