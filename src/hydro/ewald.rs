//! Periodic Rotne-Prager-Yamakawa mobility via Beenakker's Ewald sum, ported
//! faithfully from the GRPerY reference implementation (Zuk et al.,
//! github.com/pjzuk/GRPerY, `src/HYDRO/per_GRPY_polyd.f`, routines
//! ROTNE_PRAGER_TT_IJ / PER_ROTNE_PRAGER_TT_*). Equal radius here.
//!
//! Units follow GRPerY: the bare tensor is (1/8)[...] with the physical 1/(pi eta)
//! prefactor absorbed (i.e. this mobility = pi*eta * physical_mobility). Restore
//! physical units by dividing by (pi*eta). The splitting width is `sigma`
//! (alpha = 1/(sigma*sqrt2)); the periodic sum is independent of sigma, which the
//! `mobility_is_independent_of_sigma` test enforces.

use crate::hydro::mat3::Mat3;
use crate::vec3::Vec3;
use std::f64::consts::PI;

/// Parameters of the Ewald split (GRPerY convention).
#[derive(Debug, Clone, Copy)]
pub struct EwaldParams {
    pub box_l: f64,
    pub sigma: f64, // Gaussian splitting width (GRPerY uses sigma = L/sqrt(2 pi))
    pub r_cut: f64, // real-space cutoff
    pub k_max: i32, // reciprocal-space index range per axis
    pub a: f64,     // particle radius (equal)
}

/// Complementary error function (Abramowitz-Stegun 7.1.26; |error| < 1.5e-7).
#[must_use]
pub fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let y = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
        + 0.254829592)
        * t
        * (-z * z).exp();
    if x >= 0.0 { y } else { 2.0 - y }
}

/// Bare RPY translation-translation tensor (GRPerY ROTNE_PRAGER_TT_IJ),
/// equal radius `a`, separation `d` of magnitude `r`. GRPerY units (1/8 prefactor).
#[must_use]
fn bare_rpy(d: Vec3, r: f64, a: f64) -> Mat3 {
    let rr = Mat3::outer(d.scale(1.0 / r));
    let id = Mat3::identity();
    let p23a2 = (a * a + a * a) / 3.0;
    let r3 = r * r * r;
    id.scale((1.0 / r + p23a2 / r3) / 8.0)
        .add(rr.scale((1.0 / r - 3.0 * p23a2 / r3) / 8.0))
}

/// Real-space contribution coupling i to j, summed over images n (|RN|<r_cut).
/// For i==j the n=0 image is skipped. GRPerY PER_ROTNE_PRAGER_TT real lattice.
#[must_use]
pub fn real_space_block(rij: Vec3, ep: &EwaldParams, is_self: bool) -> Mat3 {
    let l = ep.box_l;
    let s = ep.sigma;
    let a2 = ep.a * ep.a;
    let two_a2 = a2 + a2;
    let mut acc = Mat3::ZERO;
    let span = (ep.r_cut / l).ceil() as i32 + 1;
    for nx in -span..=span {
        for ny in -span..=span {
            for nz in -span..=span {
                if is_self && nx == 0 && ny == 0 && nz == 0 {
                    continue;
                }
                let d = Vec3::new(
                    rij.x + f64::from(nx) * l,
                    rij.y + f64::from(ny) * l,
                    rij.z + f64::from(nz) * l,
                );
                let r = d.norm2().sqrt();
                if r > ep.r_cut || r == 0.0 {
                    continue;
                }
                let id = Mat3::identity();
                let rr = Mat3::outer(d.scale(1.0 / r));
                // erfc-screened bare RPY
                let a2a = bare_rpy(d, r, ep.a).scale(erfc(r / (s * 2f64.sqrt())));
                // Gaussian polynomial part (GRPerY PI/PRR/PRE)
                let p_i = two_a2 * (1.0 / (6.0 * s * s) + 1.0 / (3.0 * r * r));
                let p_rr = two_a2
                    * (r * r / (6.0 * s.powi(4)) - 1.0 / (3.0 * s * s) - 1.0 / (r * r))
                    + 1.0;
                let pre = (-(r * r) / (2.0 * s * s)).exp() / (4.0 * (2.0 * PI).sqrt() * s);
                let a2b = id.scale(p_i * pre).add(rr.scale(p_rr * pre));
                acc = acc.add(a2a).add(a2b);
            }
        }
    }
    acc
}

/// Reciprocal-space contribution coupling i to j (k != 0). GRPerY inverse lattice:
/// PRE*(U - (1 + sigma^2 k^2/2) k_hat k_hat), PRE = (pi/V)(1/k^2 - a^2/3) exp(-k^2 sigma^2/2),
/// times cos(k . rij).
#[must_use]
pub fn recip_space_block(rij: Vec3, ep: &EwaldParams) -> Mat3 {
    let l = ep.box_l;
    let vol = l * l * l;
    let s = ep.sigma;
    let a2 = ep.a * ep.a;
    let two_pi_l = 2.0 * PI / l;
    let id = Mat3::identity();
    let mut acc = Mat3::ZERO;
    for kx in -ep.k_max..=ep.k_max {
        for ky in -ep.k_max..=ep.k_max {
            for kz in -ep.k_max..=ep.k_max {
                if kx == 0 && ky == 0 && kz == 0 {
                    continue;
                }
                let kvec = Vec3::new(
                    f64::from(kx) * two_pi_l,
                    f64::from(ky) * two_pi_l,
                    f64::from(kz) * two_pi_l,
                );
                let k2 = kvec.norm2();
                let k = k2.sqrt();
                let kk = Mat3::outer(kvec.scale(1.0 / k));
                let pre = (PI / vol) * (1.0 / k2 - a2 / 3.0) * (-k2 * s * s / 2.0).exp();
                let tensor = id.add(kk.scale(-(1.0 + s * s * k2 / 2.0)));
                let phase = (kvec.x * rij.x + kvec.y * rij.y + kvec.z * rij.z).cos();
                acc = acc.add(tensor.scale(pre * phase));
            }
        }
    }
    acc
}

/// Assemble the periodic grand mobility (equal radius), GRPerY units. Row-major.
#[must_use]
pub fn periodic_grand_mobility(pos: &[Vec3], ep: &EwaldParams) -> Vec<f64> {
    let n = pos.len();
    let dim = 3 * n;
    let mut m = vec![0.0f64; dim * dim];
    let s = ep.sigma;
    let vol = ep.box_l.powi(3);
    let a2 = ep.a * ep.a;
    // -sigma^2 pi/(2V): the k=0 mean-backflow correction. It is the SAME for every
    // block (self and pair) - GRPerY omits it from pairs because it cancels for
    // force-balanced systems, but the raw mobility matrix needs it on all blocks
    // to be sigma-invariant.
    let backflow = -(s * s * PI / (2.0 * vol));
    // isolated self-mobility mu0 = 1/(6 a) (GRPerY units), added on the diagonal
    // (GRPerY main: C1PP += 1/(6a) U).
    let mu0 = 1.0 / (6.0 * ep.a);
    // self-only analytic n=0 replacement (the skipped real-space self image).
    let self_analytic = (a2 / (9.0 * s * s) - 1.0) / (4.0 * (2.0 * PI).sqrt() * s);
    let self_real = real_space_block(Vec3::ZERO, ep, true);
    let self_recip = recip_space_block(Vec3::ZERO, ep);
    let diag = Mat3::identity()
        .scale(mu0 + self_analytic + backflow)
        .add(self_real)
        .add(self_recip);
    for i in 0..n {
        for r in 0..3 {
            for c in 0..3 {
                m[(3 * i + r) * dim + (3 * i + c)] = diag.0[3 * r + c];
            }
        }
        for j in (i + 1)..n {
            let rij = pos[i] - pos[j];
            let block = real_space_block(rij, ep, false)
                .add(recip_space_block(rij, ep))
                .add(Mat3::identity().scale(backflow));
            for r in 0..3 {
                for c in 0..3 {
                    let v = block.0[3 * r + c];
                    m[(3 * i + r) * dim + (3 * j + c)] = v;
                    m[(3 * j + c) * dim + (3 * i + r)] = v;
                }
            }
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydro::mobility::cholesky;

    fn two_particle_box() -> Vec<Vec3> {
        vec![Vec3::new(2.0, 3.0, 4.0), Vec3::new(6.0, 5.0, 7.0)]
    }

    #[test]
    fn mobility_is_independent_of_sigma() {
        // The periodic sum must not depend on the splitting width. r_cut/k_max are
        // large enough that both sigma values are converged on each side.
        let pos = two_particle_box();
        let base = EwaldParams {
            box_l: 10.0,
            sigma: 2.0,
            r_cut: 13.0,
            k_max: 14,
            a: 1.0,
        };
        let alt = EwaldParams { sigma: 3.0, ..base };
        let m1 = periodic_grand_mobility(&pos, &base);
        let m2 = periodic_grand_mobility(&pos, &alt);
        let dim = 6;
        let mut max_abs = 0.0f64;
        for k in 0..dim * dim {
            max_abs = max_abs.max((m1[k] - m2[k]).abs());
        }
        assert!(
            max_abs < 5e-4,
            "Ewald mobility must be sigma-invariant; max diff {max_abs:.2e}"
        );
        eprintln!("sigma-invariance max diff = {max_abs:.3e}");
    }

    #[test]
    fn periodic_mobility_is_spd() {
        let pos = two_particle_box();
        let ep = EwaldParams {
            box_l: 10.0,
            sigma: 2.5,
            r_cut: 13.0,
            k_max: 14,
            a: 1.0,
        };
        let m = periodic_grand_mobility(&pos, &ep);
        assert!(cholesky(&m, 6).is_ok(), "periodic mobility not SPD");
    }
}
