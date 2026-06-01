//! Random Brownian displacements for the Ermak-McCammon step.
//!
//! Each axis is an independent Gaussian of variance `2 D dt` (standard
//! deviation `sqrt(2 D dt)`), the fluctuation-dissipation form for overdamped
//! Langevin dynamics.

use crate::vec3::Vec3;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Draw the random displacement `R ~ N(0, 2 D dt)` (per axis) for one step.
#[must_use]
pub fn brownian_displacement<R: Rng + ?Sized>(d: f64, dt: f64, rng: &mut R) -> Vec3 {
    // Standard deviation per axis: sqrt(2 D dt).
    let sigma = (2.0 * d * dt).sqrt();
    let normal = Normal::new(0.0, sigma).expect("sigma is finite and non-negative");
    Vec3::new(normal.sample(rng), normal.sample(rng), normal.sample(rng))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn brownian_displacement_has_zero_mean_and_variance_2ddt() {
        let mut rng = StdRng::seed_from_u64(42);
        let d = 0.7;
        let dt = 0.05;
        let expected = 2.0 * d * dt; // 0.07
        let n = 200_000usize;
        let (mut sum, mut sumsq) = (0.0f64, 0.0f64);
        for _ in 0..n {
            let v = brownian_displacement(d, dt, &mut rng);
            for c in [v.x, v.y, v.z] {
                sum += c;
                sumsq += c * c;
            }
        }
        let m = (3 * n) as f64;
        let mean = sum / m;
        let var = sumsq / m - mean * mean;
        assert!(mean.abs() < 2e-3, "mean should be ~0, got {mean}");
        assert!(
            (var - expected).abs() / expected < 0.02,
            "variance should be ~{expected}, got {var}"
        );
    }
}
