//! Dissociation kinetics: how long a ligand stays bound in a buried pocket
//! before escaping past the bottleneck.
//!
//! Two protocols on the same engine:
//!
//! - **residence time** (mean first-passage out of the pocket) under plain
//!   Brownian dynamics. This is the true `1/k_off`, and it follows the
//!   Kramers/Arrhenius `exp(barrier / kB T)` law.
//! - **tauRAMD** (Nunes-Alves's signature method): add a constant-magnitude
//!   random-acceleration force, periodically reoriented, that drives the ligand
//!   out far faster than equilibrium. The accelerated egress times are not the
//!   real residence times, but they *rank* them, which is what makes tauRAMD a
//!   practical predictor of relative `k_off` across a ligand series.
//!
//! This is the coarse-grained setting of her T4-lysozyme escape and
//! NiFe-hydrogenase inhibitor-dissociation studies (a ligand leaving a buried
//! pocket through a bottleneck, by multiple 3D pathways).

use crate::integrator::em_step;
use crate::potential::pocket_force;
use crate::rng::{brownian_displacement, random_unit};
use crate::vec3::Vec3;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

/// One trajectory's escape time: Brownian dynamics from the pocket centre until
/// `r >= r_b`, in time units. `accel > 0` adds a random-acceleration force of
/// that magnitude, reoriented every `reorient_steps` (the tauRAMD protocol);
/// `accel == 0` is plain BD. A trajectory that does not escape within
/// `max_steps` is censored at `max_steps * dt`.
#[allow(clippy::too_many_arguments)]
fn escape_time_one<R: rand::Rng + ?Sized>(
    barrier: f64,
    r_b: f64,
    d0: f64,
    dt: f64,
    max_steps: usize,
    accel: f64,
    reorient_steps: usize,
    rng: &mut R,
) -> f64 {
    let mut r = Vec3::ZERO;
    let rb2 = r_b * r_b;
    let mut dir = if accel > 0.0 {
        random_unit(rng)
    } else {
        Vec3::ZERO
    };
    for step in 0..max_steps {
        if accel > 0.0 && reorient_steps > 0 && step % reorient_steps == 0 {
            dir = random_unit(rng);
        }
        let mut force = pocket_force(r, barrier, r_b);
        if accel > 0.0 {
            force += dir.scale(accel);
        }
        let noise = brownian_displacement(d0, dt, rng);
        r = em_step(r, force, d0, dt, noise);
        if r.norm2() >= rb2 {
            return (step + 1) as f64 * dt;
        }
    }
    max_steps as f64 * dt // censored: did not escape in time
}

/// Mean residence time (`1/k_off`) from plain Brownian dynamics: the average
/// first-passage time out of the pocket over `replicas` independent ligands.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn mean_residence_time(
    barrier: f64,
    r_b: f64,
    d0: f64,
    dt: f64,
    max_steps: usize,
    replicas: usize,
    seed: u64,
) -> f64 {
    (0..replicas)
        .into_par_iter()
        .map(|rep| {
            let mut rng =
                StdRng::seed_from_u64(seed ^ (rep as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            escape_time_one(barrier, r_b, d0, dt, max_steps, 0.0, 0, &mut rng)
        })
        .sum::<f64>()
        / replicas as f64
}

/// Mean tauRAMD egress time: average escape time under a reoriented
/// random-acceleration force of magnitude `accel`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn tauramd_egress_time(
    barrier: f64,
    r_b: f64,
    d0: f64,
    dt: f64,
    accel: f64,
    reorient_steps: usize,
    max_steps: usize,
    replicas: usize,
    seed: u64,
) -> f64 {
    (0..replicas)
        .into_par_iter()
        .map(|rep| {
            let mut rng =
                StdRng::seed_from_u64(seed ^ (rep as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            escape_time_one(
                barrier,
                r_b,
                d0,
                dt,
                max_steps,
                accel,
                reorient_steps,
                &mut rng,
            )
        })
        .sum::<f64>()
        / replicas as f64
}

/// Record one escape trajectory: positions sampled every `stride` steps from the
/// pocket centre until the ligand crosses the bottleneck (or `max_steps`). Used
/// to visualise the multiple egress pathways out of a buried pocket.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn escape_path(
    barrier: f64,
    r_b: f64,
    d0: f64,
    dt: f64,
    accel: f64,
    reorient_steps: usize,
    max_steps: usize,
    stride: usize,
    seed: u64,
) -> Vec<Vec3> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut r = Vec3::ZERO;
    let rb2 = r_b * r_b;
    let mut path = vec![r];
    let mut dir = if accel > 0.0 {
        random_unit(&mut rng)
    } else {
        Vec3::ZERO
    };
    for step in 0..max_steps {
        if accel > 0.0 && reorient_steps > 0 && step % reorient_steps == 0 {
            dir = random_unit(&mut rng);
        }
        let mut force = pocket_force(r, barrier, r_b);
        if accel > 0.0 {
            force += dir.scale(accel);
        }
        let noise = brownian_displacement(d0, dt, &mut rng);
        r = em_step(r, force, d0, dt, noise);
        if stride > 0 && step % stride == 0 {
            path.push(r);
        }
        if r.norm2() >= rb2 {
            path.push(r);
            break;
        }
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_path_starts_at_centre_and_reaches_the_bottleneck() {
        let path = escape_path(2.0, 2.0, 1.0, 0.001, 6.0, 100, 200_000, 50, 5);
        assert_eq!(
            path[0],
            Vec3::ZERO,
            "trajectory starts at the pocket centre"
        );
        let last = *path.last().unwrap();
        assert!(
            last.norm2() >= 2.0 * 2.0 - 1e-6,
            "an accelerated trajectory should reach the bottleneck, got r^2={}",
            last.norm2()
        );
    }

    #[test]
    fn residence_time_increases_with_barrier() {
        let (r_b, d0, dt, max_steps, reps, seed) = (2.0, 1.0, 0.001, 80_000usize, 500usize, 1u64);
        let t_low = mean_residence_time(1.0, r_b, d0, dt, max_steps, reps, seed);
        let t_high = mean_residence_time(4.0, r_b, d0, dt, max_steps, reps, seed);
        assert!(
            t_high > 2.0 * t_low,
            "a higher barrier must mean a longer residence: t(1.0)={t_low:.3}, t(4.0)={t_high:.3}"
        );
    }

    #[test]
    fn tauramd_ranks_residence_and_accelerates_escape() {
        let (r_b, d0, dt, max_steps, reps, seed) = (2.0, 1.0, 0.001, 50_000usize, 800usize, 7u64);
        let (accel, reorient) = (4.0, 100usize);

        // tauRAMD egress time preserves the residence-time ordering across barriers.
        let g_low = tauramd_egress_time(0.5, r_b, d0, dt, accel, reorient, max_steps, reps, seed);
        let g_high = tauramd_egress_time(2.0, r_b, d0, dt, accel, reorient, max_steps, reps, seed);
        assert!(
            g_high > g_low,
            "tauRAMD egress must rank residence times: g(0.5)={g_low:.4}, g(2.0)={g_high:.4}"
        );

        // ...and the acceleration makes escape much faster than plain BD.
        let plain_high = mean_residence_time(2.0, r_b, d0, dt, max_steps, reps, seed);
        assert!(
            g_high < plain_high,
            "tauRAMD should accelerate escape vs plain BD: {g_high:.4} vs {plain_high:.4}"
        );
    }
}
