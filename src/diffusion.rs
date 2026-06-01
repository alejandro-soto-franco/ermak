//! Tracer diffusion runs: estimate an effective diffusion coefficient from the
//! mean squared displacement of an ensemble of Brownian walkers.
//!
//! `D_eff = MSD(t) / (6 t)` in 3D. With no interactions this recovers the bare
//! `D_0` (the analytical free-diffusion limit); among crowders it is reduced.

use crate::integrator::em_step;
use crate::rng::brownian_displacement;
use crate::vec3::Vec3;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

/// Effective diffusion coefficient of a single free tracer (no forces),
/// averaged over `replicas` independent trajectories of `steps` steps. Each
/// replica is seeded deterministically from `seed` so the run is reproducible.
#[must_use]
pub fn free_diffusion_deff(d0: f64, dt: f64, steps: usize, replicas: usize, seed: u64) -> f64 {
    // Replicas are independent and independently seeded, so the ensemble is
    // embarrassingly parallel and still reproducible for a fixed `seed`.
    let total_msd: f64 = (0..replicas)
        .into_par_iter()
        .map(|replica| {
            let mut rng =
                StdRng::seed_from_u64(seed ^ (replica as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut r = Vec3::ZERO;
            for _ in 0..steps {
                let noise = brownian_displacement(d0, dt, &mut rng);
                // Free tracer: no force, so drift vanishes and the step is the kick.
                r = em_step(r, Vec3::ZERO, 0.0, dt, noise);
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
    fn free_tracer_recovers_bare_diffusion_coefficient() {
        // Analytical limit: MSD = 6 D_0 t, so the estimated D_eff equals D_0.
        let d0 = 0.7;
        let deff = free_diffusion_deff(d0, 0.01, 2000, 4000, 1);
        let rel_err = (deff - d0).abs() / d0;
        assert!(
            rel_err < 0.04,
            "free D_eff should be ~{d0}, got {deff} (rel err {rel_err:.3})"
        );
    }
}
