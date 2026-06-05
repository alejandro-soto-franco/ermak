//! Dense grand mobility assembly (open boundary) and a hand-rolled Cholesky,
//! the CPU oracle's linear algebra. O((3N)^3); reference-scale only.

use crate::hydro::HydroSystem;
use crate::hydro::rpy::rpy_pair_equal;

/// Assemble the dense `3N x 3N` grand mobility (open boundary, equal radius
/// taken as `radius[0]` for this milestone; Task A8 generalizes per-particle).
/// Row-major.
#[must_use]
pub fn grand_mobility(sys: &HydroSystem) -> Vec<f64> {
    let n = sys.n();
    let dim = 3 * n;
    let a = sys.radius[0];
    let mut m = vec![0.0f64; dim * dim];
    for i in 0..n {
        // diagonal self block mu0_i I
        let mu0 = sys.self_mobility(i);
        for d in 0..3 {
            m[(3 * i + d) * dim + (3 * i + d)] = mu0;
        }
        for j in (i + 1)..n {
            let block = rpy_pair_equal(sys.pos[i] - sys.pos[j], a, sys.eta);
            for r in 0..3 {
                for c in 0..3 {
                    let v = block.0[3 * r + c];
                    m[(3 * i + r) * dim + (3 * j + c)] = v;
                    // symmetric: block(j,i) = block(i,j)^T, and RPY blocks are
                    // themselves symmetric, so the transpose equals the block.
                    m[(3 * j + c) * dim + (3 * i + r)] = v;
                }
            }
        }
    }
    m
}

/// In-place lower Cholesky `L` of a symmetric positive-definite row-major
/// matrix `m` (dim x dim). Returns `Err(k)` at the first non-positive pivot
/// (row k), which doubles as the positive-definiteness check.
pub fn cholesky(m: &[f64], dim: usize) -> Result<Vec<f64>, usize> {
    let mut l = vec![0.0f64; dim * dim];
    for i in 0..dim {
        for j in 0..=i {
            let mut sum = m[i * dim + j];
            for k in 0..j {
                sum -= l[i * dim + k] * l[j * dim + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(i);
                }
                l[i * dim + j] = sum.sqrt();
            } else {
                l[i * dim + j] = sum / l[j * dim + j];
            }
        }
    }
    Ok(l)
}

/// Apply the grand mobility to a per-particle force list, returning per-particle
/// drift velocities `U_i = sum_j mu_ij F_j`.
#[must_use]
pub fn apply_mobility(m: &[f64], forces: &[crate::vec3::Vec3]) -> Vec<crate::vec3::Vec3> {
    let n = forces.len();
    let dim = 3 * n;
    let mut f = vec![0.0f64; dim];
    for (i, fi) in forces.iter().enumerate() {
        f[3 * i] = fi.x;
        f[3 * i + 1] = fi.y;
        f[3 * i + 2] = fi.z;
    }
    let mut u = vec![crate::vec3::Vec3::ZERO; n];
    for i in 0..n {
        let (mut ux, mut uy, mut uz) = (0.0, 0.0, 0.0);
        for j in 0..dim {
            ux += m[(3 * i) * dim + j] * f[j];
            uy += m[(3 * i + 1) * dim + j] * f[j];
            uz += m[(3 * i + 2) * dim + j] * f[j];
        }
        u[i] = crate::vec3::Vec3::new(ux, uy, uz);
    }
    u
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec3::Vec3;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand::Rng;

    fn random_system(n: usize, seed: u64) -> HydroSystem {
        let mut rng = StdRng::seed_from_u64(seed);
        let pos = (0..n)
            .map(|_| Vec3::new(rng.gen_range(0.0..20.0), rng.gen_range(0.0..20.0), rng.gen_range(0.0..20.0)))
            .collect();
        HydroSystem { pos, radius: vec![1.0; n], charge: vec![0.0; n], eta: 1.0, kt: 1.0, box_l: None }
    }

    #[test]
    fn mobility_is_symmetric() {
        let sys = random_system(6, 1);
        let m = grand_mobility(&sys);
        let dim = 3 * sys.n();
        for i in 0..dim {
            for j in 0..dim {
                assert!((m[i * dim + j] - m[j * dim + i]).abs() < 1e-14, "asym at {i},{j}");
            }
        }
    }

    #[test]
    fn mobility_is_positive_definite_over_random_configs() {
        for seed in 0..25 {
            let sys = random_system(8, seed);
            let m = grand_mobility(&sys);
            let dim = 3 * sys.n();
            assert!(cholesky(&m, dim).is_ok(), "not SPD at seed {seed}");
        }
    }

    #[test]
    fn cholesky_reconstructs_the_matrix() {
        let sys = random_system(5, 7);
        let m = grand_mobility(&sys);
        let dim = 3 * sys.n();
        let l = cholesky(&m, dim).unwrap();
        // L L^T should equal m
        for i in 0..dim {
            for j in 0..dim {
                let mut s = 0.0;
                for k in 0..dim {
                    s += l[i * dim + k] * l[j * dim + k];
                }
                assert!((s - m[i * dim + j]).abs() < 1e-9, "LL^T mismatch at {i},{j}");
            }
        }
    }

    #[test]
    fn single_particle_drift_is_mu0_force() {
        // One particle: U = mu0 F (no cross coupling).
        let sys = HydroSystem {
            pos: vec![Vec3::ZERO], radius: vec![2.0], charge: vec![0.0],
            eta: 1.0, kt: 1.0, box_l: None,
        };
        let m = grand_mobility(&sys);
        let f = vec![Vec3::new(0.0, 0.0, 3.0)];
        let u = apply_mobility(&m, &f);
        let mu0 = sys.self_mobility(0);
        assert!((u[0].z - mu0 * 3.0).abs() < 1e-12, "U = mu0 F");
        assert!(u[0].x.abs() < 1e-15 && u[0].y.abs() < 1e-15);
    }

    #[test]
    fn equal_and_opposite_forces_drag_neighbour_along() {
        // Two particles on x; push i in +x. HI drags j in +x too (entrainment):
        // the xx cross-mobility is positive.
        let sys = HydroSystem {
            pos: vec![Vec3::ZERO, Vec3::new(5.0, 0.0, 0.0)], radius: vec![1.0, 1.0],
            charge: vec![0.0, 0.0], eta: 1.0, kt: 1.0, box_l: None,
        };
        let m = grand_mobility(&sys);
        let f = vec![Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO];
        let u = apply_mobility(&m, &f);
        assert!(u[1].x > 0.0, "neighbour entrained in +x, got {}", u[1].x);
        assert!(u[0].x > u[1].x, "driven particle faster than entrained one");
    }
}
