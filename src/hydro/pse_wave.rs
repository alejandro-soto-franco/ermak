//! CPU Spectral-Ewald reference for the wave-space (reciprocal) RPY mobility
//! (Milestone G4 step 2 Task 1). This is the particle-mesh form of the dense
//! reciprocal sum in [`crate::hydro::ewald::recip_space_block`]: spread particle
//! forces to a regular grid with a Gaussian window, forward DFT, scale by the
//! wave-space Green's function (k=0 dropped), inverse DFT, interpolate (gather)
//! back to particles. The dense reciprocal apply is the exact correctness oracle;
//! this CPU path debugs the spectral-Ewald math (window, deconvolution, k-grid
//! layout, normalization) before the GPU spread/gather/cuFFT port (Task 2).
//!
//! The transform is a brute-force separable DFT (three 1D passes), O(ng^4) in the
//! grid size, deliberately FFT-library-free: the device path uses cuFFT, this
//! reference stays a dependency-light oracle. Full periodic Gaussian (min-image
//! plus one image shell), so the only approximation is grid resolution, which the
//! convergence pin drives toward the dense Ewald sum as `ng` grows.

use crate::vec3::Vec3;
use std::f64::consts::PI;

/// Minimal complex number for the reference DFT (no external dependency).
#[derive(Debug, Clone, Copy)]
struct Cx {
    re: f64,
    im: f64,
}

impl Cx {
    const ZERO: Cx = Cx { re: 0.0, im: 0.0 };

    fn add(self, o: Cx) -> Cx {
        Cx {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }

    fn mul(self, o: Cx) -> Cx {
        Cx {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }

    fn scale(self, s: f64) -> Cx {
        Cx {
            re: self.re * s,
            im: self.im * s,
        }
    }
}

/// Parameters of the wave-space particle-mesh solve. `box_l`, `sigma`, `a` match
/// [`crate::hydro::ewald::EwaldParams`]; `eta` is the Gaussian spreading width
/// (must satisfy `eta^2 < sigma^2 / 2` for a decaying deconvolution; default
/// `eta = sigma / 2`), `ng` the cubic grid points per axis.
#[derive(Debug, Clone, Copy)]
pub struct WaveParams {
    pub box_l: f64,
    pub sigma: f64,
    pub a: f64,
    pub eta: f64,
    pub ng: usize,
    /// Gaussian spreading half-width in grid cells: a particle spreads to the
    /// `(2*support + 1)^3` nearest grid nodes (truncated Spectral Ewald). A value
    /// `>= ng/2` means the full grid (no truncation), which the CPU reference and
    /// the full-grid GPU kernels use. The truncated GPU kernels read it for the
    /// O(N P^3) window; the per-particle cost is then independent of `ng`.
    pub support: usize,
}

impl WaveParams {
    /// Full-grid defaults with `eta = sigma / 2` (residual k-space filter
    /// `exp(-k^2 sigma^2 / 4)`). `support = ng` marks the full-grid path.
    #[must_use]
    pub fn new(box_l: f64, sigma: f64, a: f64, ng: usize) -> Self {
        Self {
            box_l,
            sigma,
            a,
            eta: 0.5 * sigma,
            ng,
            support: ng,
        }
    }

    /// Truncated Spectral-Ewald parameters: an explicit spreading width `eta` and
    /// support half-width (window `(2*support + 1)` per axis). The net wave filter
    /// is independent of `eta` analytically, so a small `eta` (around the grid
    /// spacing `h = box_l/ng`) gives a compact window with negligible aliasing
    /// (`~exp(-2 pi^2 (eta/h)^2)`) and truncation (`~exp(-(support h / eta)^2 / 2)`).
    #[must_use]
    pub fn truncated(box_l: f64, sigma: f64, a: f64, ng: usize, eta: f64, support: usize) -> Self {
        Self {
            box_l,
            sigma,
            a,
            eta,
            ng,
            support,
        }
    }

    /// Whether the truncated-support GPU path applies (window smaller than the grid).
    #[must_use]
    pub fn is_truncated(&self) -> bool {
        2 * self.support + 1 < self.ng
    }
}

/// Signed Fourier mode for grid index `p` (numpy `fftfreq` layout):
/// `0, 1, ..., ng/2 - 1, -ng/2, ..., -1`.
#[inline]
fn freq(p: usize, ng: usize) -> i64 {
    #[allow(clippy::cast_possible_wrap)]
    if p < ng / 2 {
        p as i64
    } else {
        p as i64 - ng as i64
    }
}

/// One separable 3D DFT pass over a flat `ng^3` complex grid (index
/// `(x*ng + y)*ng + z`). `sign = -1.0` is the forward transform
/// `H_hat[m] = sum_g H[g] e^{-i 2 pi m.g / ng}`, `+1.0` the inverse (both
/// unnormalized, matching the cuFFT convention the GPU path will use).
fn dft3(grid: &[Cx], ng: usize, sign: f64) -> Vec<Cx> {
    // Twiddles tw[r] = e^{sign i 2 pi r / ng}, r in 0..ng.
    let tw: Vec<Cx> = (0..ng)
        .map(|r| {
            #[allow(clippy::cast_precision_loss)]
            let theta = sign * 2.0 * PI * (r as f64) / (ng as f64);
            Cx {
                re: theta.cos(),
                im: theta.sin(),
            }
        })
        .collect();
    let idx = |x: usize, y: usize, z: usize| (x * ng + y) * ng + z;

    // Pass along z, then y, then x.
    let mut cur = grid.to_vec();
    let mut next = vec![Cx::ZERO; ng * ng * ng];

    // z
    for x in 0..ng {
        for y in 0..ng {
            for m in 0..ng {
                let mut acc = Cx::ZERO;
                for q in 0..ng {
                    acc = acc.add(cur[idx(x, y, q)].mul(tw[(m * q) % ng]));
                }
                next[idx(x, y, m)] = acc;
            }
        }
    }
    std::mem::swap(&mut cur, &mut next);
    // y
    for x in 0..ng {
        for z in 0..ng {
            for m in 0..ng {
                let mut acc = Cx::ZERO;
                for q in 0..ng {
                    acc = acc.add(cur[idx(x, q, z)].mul(tw[(m * q) % ng]));
                }
                next[idx(x, m, z)] = acc;
            }
        }
    }
    std::mem::swap(&mut cur, &mut next);
    // x
    for y in 0..ng {
        for z in 0..ng {
            for m in 0..ng {
                let mut acc = Cx::ZERO;
                for q in 0..ng {
                    acc = acc.add(cur[idx(q, y, z)].mul(tw[(m * q) % ng]));
                }
                next[idx(m, y, z)] = acc;
            }
        }
    }
    next
}

/// Periodic Gaussian window `gamma(d) = (2 pi eta^2)^{-3/2} sum_images exp(-|d|^2 / 2 eta^2)`
/// for a raw displacement `d` (min-imaged, then summed over one image shell so the
/// two equidistant half-box copies are both captured; far shells are < e^{-70}).
#[inline]
fn periodic_gaussian(mut dx: f64, mut dy: f64, mut dz: f64, eta: f64, l: f64) -> f64 {
    dx -= l * (dx / l).round();
    dy -= l * (dy / l).round();
    dz -= l * (dz / l).round();
    let inv_2e2 = 1.0 / (2.0 * eta * eta);
    let mut acc = 0.0;
    for ix in -1..=1 {
        let ex = dx + f64::from(ix) * l;
        for iy in -1..=1 {
            let ey = dy + f64::from(iy) * l;
            for iz in -1..=1 {
                let ez = dz + f64::from(iz) * l;
                acc += (-(ex * ex + ey * ey + ez * ez) * inv_2e2).exp();
            }
        }
    }
    let norm = (2.0 * PI * eta * eta).powf(-1.5);
    norm * acc
}

/// Wave-space (reciprocal) RPY mobility apply via Spectral Ewald: returns
/// `U_recip_i = sum_j M_recip(r_i - r_j) F_j` for the periodic Beenakker-Ewald
/// reciprocal block of [`crate::hydro::ewald::recip_space_block`] (GRPerY units).
///
/// Pipeline (k=0 dropped): spread `F` to the grid with `gamma`, forward DFT, scale
/// each mode by `D(k) = h^3 exp(k^2 eta^2) PRE(k)(I - B(k) k_hat k_hat)`, inverse
/// DFT, gather with `gamma` (weight `h^3`). The `exp(k^2 eta^2)` deconvolves the
/// two Gaussian factors (spread and gather) so the net filter is exactly the
/// reciprocal Green's function.
///
/// # Panics
/// If `forces.len() != pos.len()` or `ng == 0`.
#[must_use]
pub fn recip_apply_pse(pos: &[Vec3], forces: &[Vec3], wp: &WaveParams) -> Vec<Vec3> {
    let n = pos.len();
    assert_eq!(forces.len(), n, "forces and positions length mismatch");
    assert!(wp.ng > 0, "grid size must be positive");
    let ng = wp.ng;
    let ng3 = ng * ng * ng;
    let l = wp.box_l;
    #[allow(clippy::cast_precision_loss)]
    let h = l / ng as f64;
    let h3 = h * h * h;
    let eta = wp.eta;
    let s = wp.sigma;
    let a2 = wp.a * wp.a;
    let vol = l * l * l;
    let two_pi_l = 2.0 * PI / l;
    let idx = |x: usize, y: usize, z: usize| (x * ng + y) * ng + z;

    // --- 1. spread each force component to the grid ---
    let mut hx = vec![Cx::ZERO; ng3];
    let mut hy = vec![Cx::ZERO; ng3];
    let mut hz = vec![Cx::ZERO; ng3];
    for gx in 0..ng {
        #[allow(clippy::cast_precision_loss)]
        let xg = h * gx as f64;
        for gy in 0..ng {
            #[allow(clippy::cast_precision_loss)]
            let yg = h * gy as f64;
            for gz in 0..ng {
                #[allow(clippy::cast_precision_loss)]
                let zg = h * gz as f64;
                let g = idx(gx, gy, gz);
                let (mut sx, mut sy, mut sz) = (0.0, 0.0, 0.0);
                for (p, f) in pos.iter().zip(forces.iter()) {
                    let w = periodic_gaussian(xg - p.x, yg - p.y, zg - p.z, eta, l);
                    sx += w * f.x;
                    sy += w * f.y;
                    sz += w * f.z;
                }
                hx[g].re = sx;
                hy[g].re = sy;
                hz[g].re = sz;
            }
        }
    }

    // --- 2. forward DFT each component ---
    let fx = dft3(&hx, ng, -1.0);
    let fy = dft3(&hy, ng, -1.0);
    let fz = dft3(&hz, ng, -1.0);

    // --- 3. scale each mode by the wave-space Green tensor D(k) ---
    let mut vx = vec![Cx::ZERO; ng3];
    let mut vy = vec![Cx::ZERO; ng3];
    let mut vz = vec![Cx::ZERO; ng3];
    for px in 0..ng {
        let mx = freq(px, ng);
        for py in 0..ng {
            let my = freq(py, ng);
            for pz in 0..ng {
                let mz = freq(pz, ng);
                if mx == 0 && my == 0 && mz == 0 {
                    continue; // k = 0 dropped
                }
                #[allow(clippy::cast_precision_loss)]
                let kx = mx as f64 * two_pi_l;
                #[allow(clippy::cast_precision_loss)]
                let ky = my as f64 * two_pi_l;
                #[allow(clippy::cast_precision_loss)]
                let kz = mz as f64 * two_pi_l;
                let k2 = kx * kx + ky * ky + kz * kz;
                let inv_k = 1.0 / k2.sqrt();
                let (hkx, hky, hkz) = (kx * inv_k, ky * inv_k, kz * inv_k); // k_hat
                let pre = (PI / vol) * (1.0 / k2 - a2 / 3.0) * (-k2 * s * s / 2.0).exp();
                let b = 1.0 + s * s * k2 / 2.0;
                let decon = h3 * (k2 * eta * eta).exp();
                // D = decon * pre * (I - b k_hat k_hat)
                let d00 = decon * pre * (1.0 - b * hkx * hkx);
                let d01 = decon * pre * (-b * hkx * hky);
                let d02 = decon * pre * (-b * hkx * hkz);
                let d11 = decon * pre * (1.0 - b * hky * hky);
                let d12 = decon * pre * (-b * hky * hkz);
                let d22 = decon * pre * (1.0 - b * hkz * hkz);
                let g = idx(px, py, pz);
                let (fkx, fky, fkz) = (fx[g], fy[g], fz[g]);
                // symmetric matrix-vector on complex components
                vx[g] = fkx.scale(d00).add(fky.scale(d01)).add(fkz.scale(d02));
                vy[g] = fkx.scale(d01).add(fky.scale(d11)).add(fkz.scale(d12));
                vz[g] = fkx.scale(d02).add(fky.scale(d12)).add(fkz.scale(d22));
            }
        }
    }

    // --- 4. inverse DFT each component ---
    let gxv = dft3(&vx, ng, 1.0);
    let gyv = dft3(&vy, ng, 1.0);
    let gzv = dft3(&vz, ng, 1.0);

    // --- 5. gather to particles (adjoint of spread, weight h^3) ---
    let mut out = vec![Vec3::ZERO; n];
    for (i, p) in pos.iter().enumerate() {
        let (mut ux, mut uy, mut uz) = (0.0, 0.0, 0.0);
        for gx in 0..ng {
            #[allow(clippy::cast_precision_loss)]
            let xg = h * gx as f64;
            for gy in 0..ng {
                #[allow(clippy::cast_precision_loss)]
                let yg = h * gy as f64;
                for gz in 0..ng {
                    #[allow(clippy::cast_precision_loss)]
                    let zg = h * gz as f64;
                    let g = idx(gx, gy, gz);
                    let w = periodic_gaussian(xg - p.x, yg - p.y, zg - p.z, eta, l);
                    ux += w * gxv[g].re;
                    uy += w * gyv[g].re;
                    uz += w * gzv[g].re;
                }
            }
        }
        out[i] = Vec3::new(ux * h3, uy * h3, uz * h3);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydro::ewald::{EwaldParams, recip_space_block};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Random non-overlapping periodic box (centres >= 2.5 apart, min-image).
    fn random_box(n: usize, l: f64, seed: u64) -> Vec<Vec3> {
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
            let ok = pos.iter().all(|q: &Vec3| {
                let mut d = p - *q;
                d = Vec3::new(
                    d.x - l * (d.x / l).round(),
                    d.y - l * (d.y / l).round(),
                    d.z - l * (d.z / l).round(),
                );
                d.norm2().sqrt() > 2.5
            });
            if ok {
                pos.push(p);
            }
        }
        pos
    }

    /// Dense reciprocal apply (the oracle): U_i = sum_j recip_space_block(r_ij) F_j.
    fn dense_recip_apply(pos: &[Vec3], forces: &[Vec3], ep: &EwaldParams) -> Vec<Vec3> {
        let n = pos.len();
        let mut out = vec![Vec3::ZERO; n];
        for i in 0..n {
            let mut acc = Vec3::ZERO;
            for j in 0..n {
                let block = recip_space_block(pos[i] - pos[j], ep);
                let f = forces[j];
                acc += Vec3::new(
                    block.0[0] * f.x + block.0[1] * f.y + block.0[2] * f.z,
                    block.0[3] * f.x + block.0[4] * f.y + block.0[5] * f.z,
                    block.0[6] * f.x + block.0[7] * f.y + block.0[8] * f.z,
                );
            }
            out[i] = acc;
        }
        out
    }

    fn max_abs_err(a: &[Vec3], b: &[Vec3]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(u, v)| {
                (u.x - v.x)
                    .abs()
                    .max((u.y - v.y).abs())
                    .max((u.z - v.z).abs())
            })
            .fold(0.0, f64::max)
    }

    #[test]
    fn pse_converges_to_dense_reciprocal() {
        let l = 10.0;
        let (sigma, a) = (2.5, 1.0);
        // Dense oracle with a converged reciprocal cutoff (exp(-k^2 s^2/2) decays).
        let ep = EwaldParams {
            box_l: l,
            sigma,
            r_cut: 13.0,
            k_max: 12,
            a,
        };
        let n = 6usize;
        let pos = random_box(n, l, 7);
        let mut rng = StdRng::seed_from_u64(99);
        let forces: Vec<Vec3> = (0..n)
            .map(|_| {
                Vec3::new(
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(-1.0..1.0),
                )
            })
            .collect();

        let u_dense = dense_recip_apply(&pos, &forces, &ep);

        let mut prev = f64::INFINITY;
        let mut finest = f64::INFINITY;
        for &ng in &[16usize, 24, 32] {
            let wp = WaveParams::new(l, sigma, a, ng);
            let u_pse = recip_apply_pse(&pos, &forces, &wp);
            let err = max_abs_err(&u_dense, &u_pse);
            eprintln!("ng={ng:>3} max_abs_err={err:.3e}");
            // Non-growth with refinement, with a round-off floor: for these well
            // separated particles the spreading aliasing is already below double
            // precision at ng=16, so the errors sit at ~1e-16 and the only
            // variation is round-off noise (the +1e-12 absorbs it).
            assert!(
                err <= prev * 1.5 + 1e-12,
                "error must not grow with grid refinement (ng={ng}: {err:.3e} vs prev {prev:.3e})"
            );
            prev = err;
            finest = err;
        }
        assert!(
            finest < 1e-4,
            "finest grid must match the dense reciprocal sum; got {finest:.3e}"
        );
    }
}
