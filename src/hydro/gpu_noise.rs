//! GPU correlated Brownian noise via Lanczos/Krylov sqrt (feature `gpu`).
//! Generates M^{1/2} xi using only mobility matrix-vector products (the G1
//! device apply), never a dense factorization. With full reorthogonalization
//! the Krylov sqrt is exact at m = dim (small N) and accurate for m << dim.
//!
//! The Cholesky factor L (used by the CPU `correlated_noise`) and the symmetric
//! root M^{1/2} here are different square roots, but both satisfy A A^T = M, so
//! both produce the fluctuation-dissipation covariance 2 kT dt M.

use crate::error::ErmakError;
use crate::hydro::ewald::EwaldParams;
use crate::hydro::gpu_ewald::GpuEwald;
use crate::vec3::Vec3;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Eigendecomposition of a small dense symmetric matrix `a` (n x n, row-major)
/// by cyclic Jacobi rotations. Returns `(eigenvalues, eigenvectors)` where
/// `eigenvectors[i*n + k]` is component `i` of the k-th eigenvector.
#[must_use]
pub fn jacobi_eigh(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut m = a.to_vec();
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..100 {
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += m[p * n + q] * m[p * n + q];
            }
        }
        if off.sqrt() < 1e-15 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (m[q * n + q] - m[p * n + p]) / (2.0 * apq);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for i in 0..n {
                    let mip = m[i * n + p];
                    let miq = m[i * n + q];
                    m[i * n + p] = c * mip - s * miq;
                    m[i * n + q] = s * mip + c * miq;
                }
                for i in 0..n {
                    let mpi = m[p * n + i];
                    let mqi = m[q * n + i];
                    m[p * n + i] = c * mpi - s * mqi;
                    m[q * n + i] = s * mpi + c * mqi;
                }
                for i in 0..n {
                    let vip = v[i * n + p];
                    let viq = v[i * n + q];
                    v[i * n + p] = c * vip - s * viq;
                    v[i * n + q] = s * vip + c * viq;
                }
            }
        }
    }
    let eig = (0..n).map(|i| m[i * n + i]).collect();
    (eig, v)
}

/// Device matvec wrapper: `M z` for a flat `3N` vector, via the G1 apply.
fn mobility_matvec(
    dev: &GpuEwald,
    pos: &[Vec3],
    ep: &EwaldParams,
    z: &[f64],
) -> Result<Vec<f64>, ErmakError> {
    let n = pos.len();
    let zf: Vec<Vec3> = (0..n)
        .map(|i| Vec3::new(z[3 * i], z[3 * i + 1], z[3 * i + 2]))
        .collect();
    let out = dev.apply_mobility_gpu(pos, &zf, ep)?;
    let mut flat = vec![0.0f64; 3 * n];
    for i in 0..n {
        flat[3 * i] = out[i].x;
        flat[3 * i + 1] = out[i].y;
        flat[3 * i + 2] = out[i].z;
    }
    Ok(flat)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// GPU mobility square-root apply: `M^{1/2} z` using the G1 device matvec.
/// Thin wrapper over [`lanczos_half_apply`].
///
/// # Errors
/// [`ErmakError::Gpu`] on any device error in the matvecs.
pub fn mobility_half_apply(
    dev: &GpuEwald,
    pos: &[Vec3],
    ep: &EwaldParams,
    z: &[f64],
    m_iters: usize,
) -> Result<Vec<f64>, ErmakError> {
    lanczos_half_apply(|v| mobility_matvec(dev, pos, ep, v), z, m_iters)
}

/// Lanczos/Krylov symmetric square-root apply driven by an arbitrary SPD
/// matrix-vector product `matvec`: returns `A^{1/2} z`. `m_iters` Krylov steps
/// with full reorthogonalization; at `m_iters = dim` the result is exact to
/// round-off. `A` enters only through `matvec`, so the same core serves the GPU
/// mobility apply and any reference apply (the statistical FD test drives it
/// with a cheap CPU matvec, the GPU matvec being already pinned to it in G1).
///
/// # Errors
/// Propagates any error returned by `matvec`.
pub fn lanczos_half_apply<F>(
    mut matvec: F,
    z: &[f64],
    m_iters: usize,
) -> Result<Vec<f64>, ErmakError>
where
    F: FnMut(&[f64]) -> Result<Vec<f64>, ErmakError>,
{
    let dim = z.len();
    let beta0 = dot(z, z).sqrt();
    if beta0 == 0.0 {
        return Ok(vec![0.0f64; dim]);
    }
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut alpha: Vec<f64> = Vec::new();
    let mut beta: Vec<f64> = Vec::new();

    let mut v: Vec<f64> = z.iter().map(|x| x / beta0).collect();
    basis.push(v.clone());
    let mut w = matvec(&v)?;
    let a0 = dot(&w, &v);
    alpha.push(a0);
    for k in 0..dim {
        w[k] -= a0 * v[k];
    }
    for bv in &basis {
        let c = dot(&w, bv);
        for k in 0..dim {
            w[k] -= c * bv[k];
        }
    }

    for _ in 1..m_iters {
        let bj = dot(&w, &w).sqrt();
        if bj < 1e-14 {
            break;
        }
        beta.push(bj);
        let vprev = v;
        v = w.iter().map(|x| x / bj).collect();
        basis.push(v.clone());
        w = matvec(&v)?;
        let aj = dot(&w, &v);
        alpha.push(aj);
        for k in 0..dim {
            w[k] -= aj * v[k] + bj * vprev[k];
        }
        for bv in &basis {
            let c = dot(&w, bv);
            for k in 0..dim {
                w[k] -= c * bv[k];
            }
        }
    }

    // assemble T_m (tridiagonal), eigendecompose, form y = sqrt(T) e_1
    let m = alpha.len();
    let mut t = vec![0.0f64; m * m];
    for i in 0..m {
        t[i * m + i] = alpha[i];
        if i + 1 < m {
            t[i * m + (i + 1)] = beta[i];
            t[(i + 1) * m + i] = beta[i];
        }
    }
    let (eig, q) = jacobi_eigh(&t, m);
    // y_j = sum_k Q[j][k] sqrt(max(lambda_k, 0)) Q[0][k]
    let mut y = vec![0.0f64; m];
    for (j, yj) in y.iter_mut().enumerate() {
        let mut acc = 0.0;
        for k in 0..m {
            acc += q[j * m + k] * eig[k].max(0.0).sqrt() * q[k];
        }
        *yj = acc;
    }
    // result = beta0 * sum_j y_j basis[j]
    let mut out = vec![0.0f64; dim];
    for (j, bv) in basis.iter().enumerate() {
        let yj = beta0 * y[j];
        for k in 0..dim {
            out[k] += yj * bv[k];
        }
    }
    Ok(out)
}

/// Correlated Brownian displacement on the GPU path:
/// `sqrt(2 kT dt) M^{1/2} xi`, `xi ~ N(0, I_{3N})`.
///
/// # Errors
/// [`ErmakError::Gpu`] on any device error.
pub fn brownian_noise_gpu<R: Rng + ?Sized>(
    dev: &GpuEwald,
    pos: &[Vec3],
    ep: &EwaldParams,
    kt: f64,
    dt: f64,
    m_iters: usize,
    rng: &mut R,
) -> Result<Vec<Vec3>, ErmakError> {
    let n = pos.len();
    let dim = 3 * n;
    let normal = Normal::new(0.0, 1.0).expect("unit normal");
    let xi: Vec<f64> = (0..dim).map(|_| normal.sample(rng)).collect();
    let half = mobility_half_apply(dev, pos, ep, &xi, m_iters)?;
    let scale = (2.0 * kt * dt).sqrt();
    Ok((0..n)
        .map(|i| {
            Vec3::new(
                scale * half[3 * i],
                scale * half[3 * i + 1],
                scale * half[3 * i + 2],
            )
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydro::ewald::periodic_grand_mobility;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn jacobi_diagonalizes_a_known_symmetric_matrix() {
        // 2x2 [[2,1],[1,2]] has eigenvalues 1 and 3.
        let (eig, _v) = jacobi_eigh(&[2.0, 1.0, 1.0, 2.0], 2);
        let mut e = eig.clone();
        e.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(
            (e[0] - 1.0).abs() < 1e-12 && (e[1] - 3.0).abs() < 1e-12,
            "eig {e:?}"
        );
    }

    fn small_box() -> (Vec<Vec3>, EwaldParams) {
        let l = 10.0;
        let ep = EwaldParams {
            box_l: l,
            sigma: 2.5,
            r_cut: 13.0,
            k_max: 6,
            a: 1.0,
        };
        let pos = vec![
            Vec3::new(2.0, 3.0, 4.0),
            Vec3::new(6.0, 5.0, 7.0),
            Vec3::new(4.0, 8.0, 2.0),
        ];
        (pos, ep)
    }

    // Requires a CUDA GPU. cargo test --features gpu -- --ignored gpu_sqrt
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn gpu_sqrt_squared_recovers_mobility() {
        // sqrt(M)(sqrt(M) z) must equal M z (the square root is exact at m = dim).
        let (pos, ep) = small_box();
        let n = pos.len();
        let dim = 3 * n;
        let dev = GpuEwald::new().expect("cuda");
        let mut rng = StdRng::seed_from_u64(5);
        let z: Vec<f64> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let half = mobility_half_apply(&dev, &pos, &ep, &z, dim).expect("half");
        let twice = mobility_half_apply(&dev, &pos, &ep, &half, dim).expect("half2");

        let zf: Vec<Vec3> = (0..n)
            .map(|i| Vec3::new(z[3 * i], z[3 * i + 1], z[3 * i + 2]))
            .collect();
        let mz = dev.apply_mobility_gpu(&pos, &zf, &ep).expect("mz");

        let mut max_abs = 0.0f64;
        for i in 0..n {
            for (a, b) in [
                (twice[3 * i], mz[i].x),
                (twice[3 * i + 1], mz[i].y),
                (twice[3 * i + 2], mz[i].z),
            ] {
                max_abs = max_abs.max((a - b).abs());
            }
        }
        eprintln!("sqrt-squared vs M: max_abs={max_abs:.3e}");
        assert!(
            max_abs < 1e-7,
            "sqrt(M)^2 must recover M; max_abs {max_abs:.3e}"
        );
    }

    /// Dense `M v` for the FD covariance test (the periodic mobility is assembled
    /// once on the host, so each Lanczos matvec is a cheap 3N x 3N apply).
    fn cpu_matvec(m: &[f64], dim: usize, v: &[f64]) -> Vec<f64> {
        (0..dim)
            .map(|a| (0..dim).map(|b| m[a * dim + b] * v[b]).sum())
            .collect()
    }

    // Host-only (no GPU): validates the spec's pin 4 (fluctuation-dissipation)
    // for the Lanczos sqrt. It drives `lanczos_half_apply` with a CPU matvec so it
    // can take the full sample count cheaply; the GPU matvec is already pinned to
    // this exact apply in G1 (max_abs 2e-16), and the GPU sqrt path is pinned by
    // `gpu_sqrt_squared_recovers_mobility`, so covariance and transport are each
    // validated by the test that fits them.
    #[test]
    fn lanczos_noise_fd_covariance() {
        let ep = EwaldParams {
            box_l: 10.0,
            sigma: 2.5,
            r_cut: 13.0,
            k_max: 6,
            a: 1.0,
        };
        let pos = vec![Vec3::new(3.0, 3.0, 3.0), Vec3::new(6.0, 3.0, 3.0)];
        let n = pos.len();
        let dim = 3 * n;
        let (kt, dt) = (1.3_f64, 0.5_f64);
        let m = periodic_grand_mobility(&pos, &ep);
        let scale = (2.0 * kt * dt).sqrt();

        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = StdRng::seed_from_u64(20_240_605);
        let samples = 400_000usize;
        let mut cov = vec![0.0f64; dim * dim];
        for _ in 0..samples {
            let xi: Vec<f64> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
            let half = lanczos_half_apply(|v| Ok(cpu_matvec(&m, dim, v)), &xi, dim).unwrap();
            let flat: Vec<f64> = half.iter().map(|x| scale * x).collect();
            for a in 0..dim {
                for b in 0..dim {
                    cov[a * dim + b] += flat[a] * flat[b];
                }
            }
        }
        let mut max_rel = 0.0f64;
        for a in 0..dim {
            for b in 0..dim {
                let est = cov[a * dim + b] / samples as f64;
                let target = 2.0 * kt * dt * m[a * dim + b];
                let denom = (2.0 * kt * dt * m[a * dim + a]).max(1e-12);
                max_rel = max_rel.max((est - target).abs() / denom);
            }
        }
        eprintln!("FD covariance max_rel={max_rel:.4}");
        assert!(max_rel < 0.03, "FD covariance off by {max_rel:.3}");
    }
}
