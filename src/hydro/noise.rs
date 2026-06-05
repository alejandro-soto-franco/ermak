//! Correlated Brownian noise for the N-body step: sqrt(2 kT dt) L xi with
//! L L^T = M. Covariance 2 kT dt M is the fluctuation-dissipation relation.

use crate::hydro::mobility::cholesky;
use crate::vec3::Vec3;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Draw the correlated displacement for one step. `l` is the precomputed lower
/// Cholesky factor of the grand mobility (3N x 3N row-major), `dim = 3N`.
#[must_use]
pub fn correlated_noise<R: Rng + ?Sized>(
    l: &[f64],
    dim: usize,
    kt: f64,
    dt: f64,
    rng: &mut R,
) -> Vec<Vec3> {
    let normal = Normal::new(0.0, 1.0).expect("unit normal");
    let xi: Vec<f64> = (0..dim).map(|_| normal.sample(rng)).collect();
    let scale = (2.0 * kt * dt).sqrt();
    let n = dim / 3;
    let mut out = vec![Vec3::ZERO; n];
    for (i, slot) in out.iter_mut().enumerate() {
        let mut comp = [0.0f64; 3];
        for (d, c) in comp.iter_mut().enumerate() {
            // lower-triangular: row (3i+d) dotted with xi up to that column
            let row = 3 * i + d;
            let mut s = 0.0;
            for k in 0..=row {
                s += l[row * dim + k] * xi[k];
            }
            *c = scale * s;
        }
        *slot = Vec3::new(comp[0], comp[1], comp[2]);
    }
    out
}

/// Convenience: factor `m` then draw. Returns `Err(k)` if `m` is not SPD.
pub fn correlated_noise_from_m<R: Rng + ?Sized>(
    m: &[f64],
    dim: usize,
    kt: f64,
    dt: f64,
    rng: &mut R,
) -> Result<Vec<Vec3>, usize> {
    let l = cholesky(m, dim)?;
    Ok(correlated_noise(&l, dim, kt, dt, rng))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydro::HydroSystem;
    use crate::hydro::mobility::grand_mobility;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn sample_covariance_matches_2_kt_dt_m() {
        // Two close particles so the cross-covariance is clearly non-zero.
        let sys = HydroSystem {
            pos: vec![Vec3::ZERO, Vec3::new(3.0, 0.0, 0.0)],
            radius: vec![1.0, 1.0],
            charge: vec![0.0, 0.0],
            eta: 1.0 / (6.0 * std::f64::consts::PI),
            kt: 1.3,
            box_l: None,
        };
        let m = grand_mobility(&sys);
        let dim = 3 * sys.n();
        let l = cholesky(&m, dim).unwrap();
        let (kt, dt) = (sys.kt, 0.5);

        let mut rng = StdRng::seed_from_u64(20240604);
        let samples = 400_000usize;
        let mut cov = vec![0.0f64; dim * dim];
        for _ in 0..samples {
            let dr = correlated_noise(&l, dim, kt, dt, &mut rng);
            let mut flat = [0.0f64; 6];
            for i in 0..2 {
                flat[3 * i] = dr[i].x;
                flat[3 * i + 1] = dr[i].y;
                flat[3 * i + 2] = dr[i].z;
            }
            for a in 0..dim {
                for b in 0..dim {
                    cov[a * dim + b] += flat[a] * flat[b];
                }
            }
        }
        // sample covariance vs target 2 kT dt M
        let mut max_rel = 0.0f64;
        for a in 0..dim {
            for b in 0..dim {
                let est = cov[a * dim + b] / samples as f64;
                let target = 2.0 * kt * dt * m[a * dim + b];
                let denom = (2.0 * kt * dt * m[a * dim + a]).max(1e-12);
                max_rel = max_rel.max((est - target).abs() / denom);
            }
        }
        assert!(max_rel < 0.03, "FD covariance off by {max_rel:.3} (rel)");
    }
}
